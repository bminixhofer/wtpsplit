import argparse
import math
import random
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import transformers
from datasets import Dataset
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

import wandb
from wtpsplit.models import SubwordXLMForTokenClassification
from wtpsplit.utils import Constants

parser = argparse.ArgumentParser()
parser.add_argument("--block_size", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--lim_lookahead", type=bool, default=False)
parser.add_argument("--upsample_non_whitespace", type=bool, default=False)
parser.add_argument("--without_pretraining", type=bool, default=False)
parser.add_argument("--corruption_in_pretraining", type=bool, default=False)
parser.add_argument("--new_tokenizer", type=bool, default=False)
parser.add_argument("--upsampling_in_pretraining", type=bool, default=False)
args = parser.parse_args()


data_path = "data/all_data.pth"
all_data = torch.load(data_path)

block_size = args.block_size

train_sentences = defaultdict(lambda: defaultdict(list))
test_sentences = defaultdict(lambda: defaultdict(list))

for lang_code in tqdm(all_data, desc="Loading train/dev data"):

    if (
        "ud" in all_data[lang_code]["sentence"]
        and all_data[lang_code]["sentence"]["ud"]["meta"]["train_data"] is not None
    ):
        print(f"Found UD data for {lang_code}")

        train_data = all_data[lang_code]["sentence"]["ud"]["meta"]["train_data"]
        train_sentences[lang_code]["all"].extend(train_data)

        train_data = all_data[lang_code]["sentence"]["ud-corrupted"]["meta"]["train_data"]
        train_sentences[lang_code]["all"].extend(train_data)

    elif (
        "opus100" in all_data[lang_code]["sentence"]
        and all_data[lang_code]["sentence"]["opus100"]["meta"]["train_data"] is not None
    ):

        print(f"Found Opus100 data for {lang_code}")

        train_data = all_data[lang_code]["sentence"]["opus100"]["meta"]["train_data"][:10000]
        train_sentences[lang_code]["all"].extend(train_data)

        train_data = all_data[lang_code]["sentence"]["opus100-corrupted"]["meta"]["train_data"][:10000]
        train_sentences[lang_code]["all"].extend(train_data)
    else:
        print(f"No data found for {lang_code}")

    for dataset in all_data[lang_code]["sentence"]:

        test_data = all_data[lang_code]["sentence"][dataset]["data"]
        test_sentences[dataset][lang_code].extend(test_data[:100])


tokenizer_checkpoint = "xlm-roberta-base"

if args.without_pretraining:
    model_checkpoint = "xlm-roberta-base"
elif args.upsampling_in_pretraining:
    model_checkpoint = "data/models/xlmr-3l-v3_look48_snW4"
elif args.num_layers == 3:
    if args.lim_lookahead:
        model_checkpoint = "data/models/xlmr-3l-v3_look48-NEW"
    else:
        model_checkpoint = "data/models/xlmr-3l-v3"
elif args.num_layers == 6:
    if args.lim_lookahead:
        model_checkpoint = "data/models/xlmr-6l-v3_look48-OLD"
    else:
        model_checkpoint = "data/models/xlmr-6l-v3"
elif args.num_layers == 12:
    if args.corruption_in_pretraining:
        model_checkpoint = "data/models/xlmr-12l-v3_lc0.25-mix2"
    elif args.new_tokenizer:
        model_checkpoint = "data/models/xlmr-12l-v3-NEW"
    else:
        model_checkpoint = "data/models/xlmr-12l-v3-OLD"
else:
    raise ValueError("Invalid number of layers. Valid values are 3, 6, 12.")

print(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

model = SubwordXLMForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=1,
    ignore_mismatched_sizes=True,
)

print("Model loaded")


def tokenize_and_get_labels(sentences, separator):

    joined_sentence = ""
    sentence_start_positions = []
    current_position = 0

    for sentence in sentences:
        if joined_sentence:
            joined_sentence += separator
            current_position += len(separator)
        start_position = current_position
        joined_sentence += sentence
        current_position += len(sentence)
        sentence_start_positions.append(start_position + len(sentence) - 1)

    tokenized_input = tokenizer(
        joined_sentence,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )

    tokens = tokenized_input.tokens()
    offsets = tokenized_input["offset_mapping"]
    sentence_ending_labels = [0] * len(tokens)

    sentence_ending_labels[-1] = 1
    sentence_index = 0

    for i in range(len(offsets)):
        if offsets[i][0] > sentence_start_positions[sentence_index]:
            sentence_ending_labels[i - 1] = 1
            sentence_index += 1

    input_ids = [0] + tokenized_input["input_ids"] + [2]
    labels = [0] + sentence_ending_labels + [0]

    return input_ids, labels


def pack_sentences(input_data_dict, block_size, by_dataset_too=False):

    if not by_dataset_too:
        packed_data = defaultdict(lambda: {"input_ids": [], "attention_mask": [], "labels": []})
    else:
        packed_data = defaultdict(lambda: defaultdict(lambda: {"input_ids": [], "attention_mask": [], "labels": []}))

    for dataset_name in tqdm(input_data_dict):
        for lang_code, sentences in input_data_dict[dataset_name].items():

            separator = Constants.SEPARATORS.get(lang_code, " ")

            token_count, one_block_sentences = 0, []

            for sentence in sentences:
                if not sentence or sentence.isnumeric():
                    continue

                # TODO change this to tokenize in one go for efficiency
                num_sentence_tokens = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])

                if token_count + num_sentence_tokens < block_size - 4:
                    one_block_sentences.append(sentence)
                    token_count += num_sentence_tokens
                else:
                    if one_block_sentences:
                        input_ids, labels = tokenize_and_get_labels(one_block_sentences, separator)

                        num_to_pad = block_size - len(input_ids)
                        attention_mask = [1] * len(input_ids) + [0] * num_to_pad
                        input_ids += [tokenizer.pad_token_id] * num_to_pad
                        labels += [-100] * num_to_pad

                        assert len(input_ids) == block_size, len(input_ids)
                        assert len(input_ids) == len(labels), (
                            len(input_ids),
                            len(labels),
                        )

                        if not by_dataset_too:
                            packed_data[lang_code]["input_ids"].append(input_ids)
                            packed_data[lang_code]["attention_mask"].append(attention_mask)
                            packed_data[lang_code]["labels"].append(labels)
                        else:
                            packed_data[dataset_name][lang_code]["input_ids"].append(input_ids)
                            packed_data[dataset_name][lang_code]["attention_mask"].append(attention_mask)
                            packed_data[dataset_name][lang_code]["labels"].append(labels)

                    if num_sentence_tokens > block_size - 4:
                        one_block_sentences = []
                        token_count = 0
                    else:
                        one_block_sentences = [sentence]
                        token_count = num_sentence_tokens

            if one_block_sentences:
                input_ids, labels = tokenize_and_get_labels(one_block_sentences, separator)

                num_to_pad = block_size - len(input_ids)
                attention_mask = [1] * len(input_ids) + [0] * num_to_pad
                input_ids += [tokenizer.pad_token_id] * num_to_pad
                labels += [-100] * num_to_pad

                assert len(input_ids) == block_size, len(input_ids)
                assert len(input_ids) == len(labels), (len(input_ids), len(labels))

                if not by_dataset_too:
                    packed_data[lang_code]["input_ids"].append(input_ids)
                    packed_data[lang_code]["attention_mask"].append(attention_mask)
                    packed_data[lang_code]["labels"].append(labels)
                else:
                    packed_data[dataset_name][lang_code]["input_ids"].append(input_ids)
                    packed_data[dataset_name][lang_code]["attention_mask"].append(attention_mask)
                    packed_data[dataset_name][lang_code]["labels"].append(labels)

            if not by_dataset_too:
                assert len(packed_data[lang_code]["input_ids"]) == len(packed_data[lang_code]["labels"])
            else:
                assert len(packed_data[dataset_name][lang_code]["input_ids"]) == len(
                    packed_data[dataset_name][lang_code]["labels"]
                )

    return packed_data


packed_train_data = pack_sentences(train_sentences, block_size)

packed_test_data = pack_sentences(test_sentences, block_size, by_dataset_too=True)

test_dataset = {dataset_name: defaultdict(dict) for dataset_name in packed_test_data}

for dataset_name in packed_test_data:
    for lang_code in packed_test_data[dataset_name]:
        test_dataset[dataset_name][lang_code] = Dataset.from_dict(packed_test_data[dataset_name][lang_code])

if args.lim_lookahead:
    lookahead = 48
else:
    lookahead = 512

experiment_name = model_checkpoint.split("/")[-1]

experiment_name += f"-FT-{args.num_layers}L-{args.block_size}BS-{lookahead}LA"

if args.upsampling_in_pretraining:
    experiment_name += "-upsample_nW4_in_WtP"

if args.upsample_non_whitespace:
    experiment_name += "-upsample_nW4_in_FT"


def compute_prf(true_values, predicted_values):

    TP = np.sum((predicted_values == 1) & (true_values == 1))
    FP = np.sum((predicted_values == 1) & (true_values == 0))
    FN = np.sum((predicted_values == 0) & (true_values == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(p):

    predictions, labels = p

    predictions = np.reshape(predictions, (-1,))
    labels = np.reshape(labels, (-1,))

    predictions = sigmoid_array(predictions)

    predictions = predictions[labels != -100]
    labels = labels[labels != -100]

    thresholds = np.concatenate(
        [
            np.arange(0.0000001, 0.000001, 0.0000001),
            np.arange(0.000001, 0.00001, 0.000001),
            np.arange(0.00001, 0.0001, 0.00001),
            np.arange(0.0001, 0.001, 0.0001),
            np.arange(0.001, 0.01, 0.001),
            np.arange(0.01, 0.1, 0.01),
            np.arange(0.1, 1, 0.1),
        ]
    )

    precision_best = 0
    recall_best = 0
    f1_best = 0
    threshold_best = 0

    for threshold in thresholds[::-1]:
        preds = (predictions > threshold).astype(int)

        precision, recall, f1 = compute_prf(labels, preds)

        if f1 > f1_best:
            precision_best = precision
            recall_best = recall
            f1_best = f1
            threshold_best = threshold

    output_dict = {
        "precision": precision_best,
        "recall": recall_best,
        "f1": f1_best,
        "threshold": threshold_best,
    }

    return output_dict


class MultiDatasetEvalCallback(TrainerCallback):
    def __init__(self, eval_datasets):
        self.eval_datasets = eval_datasets

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.eval_steps == 0:

            for dataset_name in self.eval_datasets:
                for lang_code, eval_dataset in self.eval_datasets[dataset_name].items():

                    metrics = trainer.evaluate(eval_dataset)

                    for metric, result in metrics.items():
                        wandb.log(
                            {
                                f"eval/{dataset_name}/{lang_code}/{metric}": result,
                                "train/global_step": state.global_step,
                            }
                        )


multi_dataset_eval_callback = MultiDatasetEvalCallback(test_dataset)

if not args.upsample_non_whitespace:

    train_datasets = [Dataset.from_dict(data) for lang_code, data in packed_train_data.items()]
else:
    train_datasets = []

    for lang_code, data in packed_train_data.items():
        if Constants.SEPARATORS.get(lang_code, " ") == "":
            train_datasets.extend([Dataset.from_dict(data)] * 20)
        else:
            train_datasets.append(Dataset.from_dict(data))

random.shuffle(train_datasets)


train_datasets = ConcatDataset(train_datasets)


run = wandb.init(project="WtP-FT", entity="igorsterner")
wandb.run.name = experiment_name

args = TrainingArguments(
    output_dir=Path("data/models") / experiment_name,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=250,
    report_to="wandb",
    learning_rate=3e-5,
    warmup_steps=500,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    weight_decay=0.01,
    push_to_hub=False,
    save_total_limit=1,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=False,
    max_steps=20000,
)


class RoundRobinSampler:

    def __init__(self, samplers: Sequence[Iterable], reinit: bool = False):

        self.samplers = samplers
        self.reinit = reinit

    def __iter__(self):
        iterators = [iter(sampler) for sampler in self.samplers]

        for i in cycle(range(len(iterators))):
            it = iterators[i]

            try:
                yield next(it)

            except StopIteration:
                if not self.reinit:
                    break

                # re-initialize the iterator
                it = iter(self.samplers[i])
                iterators[i] = it
                yield next(it)


def get_subset(length: int, i: int, k: int, offset: int = 0) -> Tuple[int, int]:
    assert i < k
    s = math.ceil(length / k)  # size of one split
    start = i * s
    end = min((i + 1) * s, length)
    return offset + start, offset + end


class DistributedRoundRobinBatchSampler:

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        rank: int,
        num_replicas: int,
        drop_last: bool = False,
        seed: int = 0,
        shuffle: bool = True,
        reinit: bool = False,
    ):
        self.lengths = lengths
        offsets = [sum(lengths[:i]) for i in range(len(lengths))]
        self.ranges = [get_subset(length, rank, num_replicas, offset) for offset, length in zip(offsets, lengths)]
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.reinit = reinit
        self.batch_size = batch_size
        self.batch_start = 0

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        batch_samplers = [
            BatchSampler(
                (SubsetRandomSampler(range(start, end), generator=g) if self.shuffle else range(start, end)),
                self.batch_size,
                self.drop_last,
            )
            for (start, end) in self.ranges
        ]

        sampler = RoundRobinSampler(batch_samplers, reinit=self.reinit)
        return iter(sampler)

    def __len__(self):
        return min(length for length in self.lengths) // self.batch_size


class CustomTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset

        if isinstance(dataset, ConcatDataset):
            sizes = [len(ds) for ds in dataset.datasets]
        else:
            sizes = [len(dataset)]

        loader = DataLoader(
            dataset,
            batch_sampler=DistributedRoundRobinBatchSampler(
                lengths=sizes,
                batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                rank=self.args.process_index,
                num_replicas=self.args.world_size,
                seed=self.args.seed,
                reinit=True,
            ),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            collate_fn=self.data_collator,
        )
        return loader


trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_datasets,  # Now it's a concatenated dataset
    eval_dataset=None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[multi_dataset_eval_callback],
)

trainer.train()
