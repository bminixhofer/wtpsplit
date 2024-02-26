from dataclasses import dataclass
import logging
import sys
import os
import copy
from typing import List
from adapters import AdapterArguments
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed
from wtpsplit.train.evaluate import evaluate_sentence
from wtpsplit.train.adaptertrainer import AdapterTrainer
from wtpsplit.utils import Constants, LabelArgs, get_label_dict, get_subword_label_dict
from wtpsplit.train.utils import Model
from wtpsplit.train.train import setup_logging, collate_fn
from wtpsplit.models import SubwordXLMForTokenClassification, SubwordXLMConfig
from tokenizers import AddedToken

import adapters
import datasets
import numpy as np
import math
from collections import Counter
import torch
import random
import wandb
from glob import glob
from functools import partial

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Args:
    model_name_or_path: str
    base_model: str = "xlm-roberta-base"
    shuffle: bool = True
    text_path: str = "data/eval.pth"
    include_languages: List[str] = None
    preprocessing_num_workers: int = 1
    block_size: int = 512
    overflow_size: int = 16
    eval_stride: int = 256
    loss_margin: float = 0.5
    pack_samples: bool = False
    one_sample_per_line: bool = False
    use_loss_weights: bool = False
    do_sentence_training: bool = True
    do_auxiliary_training: bool = True
    aux_training_weight: float = 1.0
    ignore_non_hyphen: bool = False
    non_punctuation_sample_ratio: float = None
    adapter_warmup_steps: int = 0
    adapter_lr_multiplier: float = 1.0
    text_column: str = "text"

    # NEW PARAMS
    use_subwords: bool = False
    freeze_classifier: bool = False
    clf_from_scratch: bool = False


def main():
    parser = HfArgumentParser([Args, TrainingArguments, LabelArgs, AdapterArguments])
    if sys.argv[1].endswith(".json"):
        (args, training_args, label_args, adapter_args) = parser.parse_json_file(sys.argv[1])
        wandb_name = training_args.output_dir
    else:
        (args, training_args, label_args, adapter_args) = parser.parse_args_into_dataclasses()
        wandb_name = None

    setup_logging(training_args)
    set_seed(training_args.seed)

    num_labels = Constants.AUX_OFFSET + (
        (1 + len(Constants.PUNCTUATION_CHARS)) if args.do_auxiliary_training or label_args.use_auxiliary else 0
    )
    config = SubwordXLMConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
    )

    # since we pre-tokenize, running multiple epochs would iterate over data in same order
    # hence, we duplicate & shuffle train data sentences in prepare_dataset
    # and set num_train_epochs to 1 --> simulate multiple epochs, each with different sentence order
    num_train_epochs = training_args.num_train_epochs

    training_args.num_train_epochs = 1
    training_args.evaluation_strategy = "steps"

    def prepare_dataset(
        data,
        num_workers=1,
        include_languages=None,
        dataset_name="ud",
        shuffle=False,
        split="train",
    ):
        # maybe we use more than 1 lang later at once.
        with training_args.main_process_first():
            for lang in include_languages:
                if split == "train":
                    dataset = data[lang]["sentence"][dataset_name]["meta"]["train_data"]
                elif split == "valid":
                    dataset = data[lang]["sentence"][dataset_name]["data"]
                data_list = []
                if dataset is None:
                    return None
                for sample in dataset:
                    ends_with_punctuation = sample.endswith(tuple(Constants.PUNCTUATION_CHARS))
                    data_list.append(
                        {
                            args.text_column: sample + "\n" if len(sample) > 0 and sample[-1] != "\n" else sample,
                            "lang": lang,
                            "ends_with_punctuation": ends_with_punctuation,
                        }
                    )
                dataset = datasets.Dataset.from_list(data_list)
                with training_args.main_process_first():
                    logger.warning(f"Loaded {len(dataset)} examples for {lang} {dataset_name} {split} dataset.")

        if include_languages is not None:
            include_languages = set(include_languages)

            dataset = dataset.filter(
                lambda example: example["lang"] in include_languages,
                num_proc=args.preprocessing_num_workers,
            )
            with training_args.main_process_first():
                logger.warning(f"Filtered to {len(dataset)} examples.")

        if shuffle:
            # create n_epochs copies of the dataset and shuffle them individually
            dataset = datasets.concatenate_datasets([dataset.shuffle(seed=i) for i in range(num_train_epochs)])

            with training_args.main_process_first():
                logger.warning(f"Shuffled dataset to {len(dataset)} examples.")

        # very likely not relevant / used only for the compound part
        if args.ignore_non_hyphen:
            with training_args.main_process_first():
                dataset = dataset.filter(
                    lambda sample: any(c in sample[args.text_column] for c in label_args.hyphen_chars),
                    num_proc=args.preprocessing_num_workers,
                )
                with training_args.main_process_first():
                    logger.info(f"Filtered to {len(dataset)} examples.")

        # "punctuation-specific sampling" in the paper
        if args.non_punctuation_sample_ratio is not None:
            languages_without_punctuation = {
                lang_code
                for lang_code in Constants.LANGINFO.index
                if Constants.LANGINFO.loc[lang_code, "no_punctuation"]
            }

            def drop_some_non_punctuation_samples(examples):
                include_indices = set(
                    np.where([lang_code not in languages_without_punctuation for lang_code in examples["lang"]])[0]
                )
                punctuation_indices = {
                    i for i in np.where(examples["ends_with_punctuation"])[0] if i in include_indices
                }

                target_n_non_punct = int(
                    (len(punctuation_indices) * args.non_punctuation_sample_ratio)
                    / (1 - args.non_punctuation_sample_ratio)
                )
                n_drop = (len(include_indices) - len(punctuation_indices)) - target_n_non_punct

                out = [True for _ in range(len(examples["ends_with_punctuation"]))]

                if n_drop <= 0:
                    return out
                drop_indices = np.random.choice(
                    list(include_indices - punctuation_indices),
                    n_drop,
                    replace=False,
                )

                for i in drop_indices:
                    out[i] = False

                return out

            with training_args.main_process_first():
                dataset = dataset.filter(
                    drop_some_non_punctuation_samples,
                    batched=True,
                    batch_size=1_000_000,
                    num_proc=num_workers,
                )

        def tokenize_texts(examples):
            # do not return CLS and SEP token here
            # there should only be 1 of these per block later, not multiple
            # we still can't use return_special_tokens=False since we need the \n token later for the labels
            tokenized = tokenizer(examples[args.text_column], verbose=False)
            return {"input_ids": [example[1:-1] for example in tokenized["input_ids"]]}

        # similar to group_texts in huggingface's run_clm.py / run_mlm.py: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
        def group_texts(examples):
            all_input_blocks = []
            all_input_block_lengths = []
            all_langs = []

            def maybe_pad(text):
                if args.pack_samples:
                    padding = model.backbone.config.downsampling_rate - (len(text) % model.backbone.downsampling_rate)
                    if padding == model.backbone.downsampling_rate:
                        padding = 0

                    text += chr(0) * padding

                return text

            for current_lang in set(examples["lang"]):
                if not args.use_subwords:
                    lang_texts = [
                        maybe_pad(text)
                        for text, lang in zip(examples["input_ids"], examples["lang"])
                        if lang == current_lang
                    ]
                else:
                    # only retain current_lang examples (all columns)
                    lang_subwords = [
                        subwords
                        for subwords, lang in zip(examples["input_ids"], examples["lang"])
                        if lang == current_lang
                    ]
                    # filter out some special tokens
                    # from html tags, mostly in Latin, Thai & Korean
                    lang_subwords = [
                        [subword for subword in subwords if subword not in special_tokens_ids]
                        for subwords in lang_subwords
                    ]
                # pack_samples used for the compound part, so irrelevant
                if args.pack_samples:
                    if args.use_subwords:
                        raise NotImplementedError
                    blocks = []
                    block_ids = []

                    current_block = ["", []]

                    for i, text in enumerate(lang_texts):
                        if len(text) > args.block_size:
                            continue

                        current_block[0] += text
                        current_block[1] += [i] * len(text)

                        if i + 1 < len(lang_texts) and len(current_block[0]) + len(lang_texts[i + 1]) > args.block_size:
                            padding = args.block_size - len(current_block[0])

                            current_block[0] += chr(0) * padding
                            current_block[1] += [i] * padding
                            blocks.append(current_block[0])
                            block_ids.append(current_block[1])

                            current_block = ["", []]

                    if len(current_block[0]) > 0:
                        padding = args.block_size - len(current_block[0])

                        current_block[0] += chr(0) * padding
                        current_block[1] += [i] * padding
                        blocks.append(current_block[0])
                        block_ids.append(current_block[1])
                else:
                    if not args.use_subwords:
                        concatenated_texts = "".join(lang_texts)
                        concatenated_ids = [i for i, text in enumerate(lang_texts) for _ in text]
                    else:
                        # concatenate token lists
                        concatenated_texts = [item for sublist in lang_subwords for item in sublist]
                        concatenated_ids = [i for i, subwords in enumerate(lang_subwords) for _ in subwords]

                    total_length = len(concatenated_texts)

                    best_length = math.ceil(total_length / args.block_size) * args.block_size + args.overflow_size
                    while best_length > total_length:
                        best_length -= args.block_size

                    if best_length < 0:
                        continue

                    concatenated_texts = concatenated_texts[:best_length]
                    concatenated_ids = concatenated_ids[:best_length]

                    blocks = [
                        concatenated_texts[i : i + args.block_size + args.overflow_size]
                        for i in range(0, best_length - args.block_size, args.block_size)
                    ]
                    block_ids = [
                        concatenated_ids[i : i + args.block_size + args.overflow_size]
                        for i in range(0, best_length - args.block_size, args.block_size)
                    ]

                block_langs = [current_lang] * len(blocks)

                all_input_blocks.extend(blocks)
                all_input_block_lengths.extend([list(Counter(ids).values()) for ids in block_ids])
                all_langs.extend(block_langs)

            return {
                "input_ids": all_input_blocks,
                "block_lengths": all_input_block_lengths,
                "lang": all_langs,
            }

        if args.pack_samples:
            assert not args.one_sample_per_line

        if args.use_subwords:
            with training_args.main_process_first():
                dataset = dataset.map(
                    tokenize_texts,
                    batched=True,
                    num_proc=num_workers,
                    remove_columns=[args.text_column],
                )
        else:
            # this is no longer used and would cause an error otherwise
            with training_args.main_process_first():
                dataset = dataset.rename_column(args.text_column, "input_ids")

        if not args.one_sample_per_line:
            with training_args.main_process_first():
                dataset = dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=num_workers,
                    # a bit hacky but oh well, only drop if sentence
                    remove_columns=["ends_with_punctuation"] if args.text_column == "text" else [],
                )

        return dataset

    with training_args.main_process_first():
        data = torch.load(
            args.text_path,
        )

    if not args.include_languages:
        args.include_languages = list(data.keys())  # use all

    # 1 wandb run for all language-dataset combinations
    if "wandb" in training_args.report_to and training_args.process_index == 0:
        wandb.init(name=wandb_name, project="sentence-peft")
        wandb.config.update(args)
        wandb.config.update(training_args)
        wandb.config.update(label_args)

        for file in glob(os.path.join(os.path.dirname(__file__), "*.py")):
            wandb.save(os.path.abspath(file), policy="now")

    for lang in data.keys():
        if lang in args.include_languages:
            for dataset_name in data[lang]["sentence"].keys():
                # do model stuff here; otherwise, head params would be overwritten every time
                backbone = SubwordXLMForTokenClassification.from_pretrained(
                    args.model_name_or_path, config=config, ignore_mismatched_sizes=True
                )
                backbone.config.base_model = args.base_model

                # setup adapters
                model_type = backbone.config.model_type
                # adapters need xlm-roberta as model type.
                backbone.config.model_type = "xlm-roberta"  # needed for adapter setup
                adapters.init(backbone)
                # reset model type (used later)
                backbone.config.model_type = model_type

                tokenizer = AutoTokenizer.from_pretrained(args.base_model)
                # needed since we create labels in collate_fn based on tokens
                tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
                custom_token_id = tokenizer.convert_tokens_to_ids("\n")
                # used later to filter out special tokens
                special_tokens_ids = set(tokenizer.all_special_ids)
                special_tokens_ids.discard(custom_token_id)

                model = Model(
                    backbone,
                    loss_margin=args.loss_margin,
                    use_loss_weights=args.use_loss_weights,
                    do_sentence_training=args.do_sentence_training,
                    do_auxiliary_training=args.do_auxiliary_training,
                    aux_training_weight=args.aux_training_weight,
                )

                with training_args.main_process_first():
                    valid_dataset = prepare_dataset(
                        data=data,
                        num_workers=1,
                        include_languages=[lang],
                        dataset_name=dataset_name,
                        shuffle=False,
                        split="valid",
                    )
                    logger.warning(f"Valid ds for {lang} {dataset_name} has {len(valid_dataset)} examples.")

                    train_dataset = prepare_dataset(
                        data=data,
                        num_workers=args.preprocessing_num_workers,
                        include_languages=[lang],
                        dataset_name=dataset_name,
                        shuffle=args.shuffle,
                        split="train",
                    )
                    if train_dataset is None or valid_dataset is None:
                        logger.warning(f"Skipping {lang} {dataset_name} due to missing data.")
                        continue
                    logger.warning(f"Train ds for {lang} {dataset_name} has {len(train_dataset)} examples.")

                # eval every actual epoch, based on steps
                training_args.eval_steps = (
                    len(train_dataset)
                    // (
                        training_args.per_device_train_batch_size
                        * training_args.gradient_accumulation_steps
                        * num_train_epochs
                    )
                ) + 1

                # print some samples from the dataset
                count = 0
                while count < 1:
                    index = random.choice(range(len(train_dataset)))
                    sample = train_dataset[index]

                    logger.warning(f"Sample {index} of the training set: {sample}.")
                    if tokenizer:
                        logger.warning(tokenizer.decode(sample["input_ids"]))
                    count += 1

                def compute_metrics(trainer):
                    metrics = {}
                    eval_data = data[lang]["sentence"][dataset_name]["data"]

                    model = trainer._wrap_model(trainer.model, training=False)

                    with training_args.main_process_first():
                        score, info = evaluate_sentence(
                            lang,
                            eval_data,
                            model,
                            stride=64,
                            block_size=512,  ## TODO: change to args version x2?
                            batch_size=training_args.per_device_eval_batch_size,
                        )
                        metrics[f"{lang}_{dataset_name}_pr_auc"] = score
                        metrics[f"{lang}_{dataset_name}_f1"] = info["f1"]
                        metrics[f"{lang}_{dataset_name}_f1_best"] = info["f1_best"]
                        metrics[f"{lang}_{dataset_name}_threshold_best"] = info["threshold_best"]

                        return metrics

                label_dict = (
                    get_subword_label_dict(label_args, tokenizer) if args.use_subwords else get_label_dict(label_args)
                )

                # init new adapter
                model.backbone.add_adapter(
                    "text", config=adapter_args.adapter_config, set_active=True, overwrite_ok=True
                )
                model.backbone.train_adapter("text")
                with training_args.main_process_first():
                    logger.warning(model.backbone.adapter_summary())

                if args.freeze_classifier:
                    for n, p in model.backbone.named_parameters():
                        if "classifier" in n:
                            p.requires_grad = False
                if args.clf_from_scratch:
                    model.backbone.classifier = torch.nn.Linear(model.backbone.config.hidden_size, num_labels)

                trainer = AdapterTrainer(
                    model,
                    training_args,
                    train_dataset=train_dataset,
                    eval_dataset=valid_dataset,
                    compute_metrics=compute_metrics,
                    data_collator=partial(
                        collate_fn,
                        args=args,
                        label_args=label_args,
                        label_dict=label_dict,
                        tokenizer=tokenizer,
                    ),
                    logging_suffix=f"{lang}_{dataset_name}",
                )
                trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
                with training_args.main_process_first():
                    if not os.path.exists(os.path.join(training_args.output_dir, dataset_name, lang)):
                        os.makedirs(os.path.join(training_args.output_dir, dataset_name, lang))
                    save_model = copy.deepcopy(model.backbone)
                    save_model = save_model.to("cpu")
                    save_model.to("cpu").save_adapter(
                        adapter_name="text",
                        save_directory=os.path.join(training_args.output_dir, dataset_name, lang),
                        with_head=True,
                    )

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # try:
    main()
    # except Exception:
    #     # extype, value, tb = sys.exc_info()
    #     # tb.print_exc()
    #     # pdb.post_mortem(tb)
    #     pass
