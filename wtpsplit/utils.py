import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from cached_property import cached_property
from mosestokenizer import MosesTokenizer
from transformers import AutoTokenizer

# same as in CANINE
PRIMES = [31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223]

logger = logging.getLogger(__name__)


class ConstantsClass:
    NEWLINE_INDEX = 0
    AUX_OFFSET = 1
    DEFAULT_PUNCTUATION_FILE = "punctuation.txt"
    _PUNCTUATION_FILE = "punctuation.txt"

    @classmethod
    def set_punctuation_file(cls, file_name):
        cls._PUNCTUATION_FILE = file_name

    @cached_property
    def ROOT_DIR(self):
        return Path(os.path.abspath(os.path.join(os.path.dirname(__file__))))

    @cached_property
    def CACHE_DIR(self):
        CACHE_DIR = self.ROOT_DIR / ".cache"
        CACHE_DIR.mkdir(exist_ok=True)
        return CACHE_DIR

    @cached_property
    def LANGINFO(self):
        return pd.read_csv(os.path.join(self.ROOT_DIR, "data", "language_info.csv"), index_col=0)

    @property
    def PUNCTUATION_CHARS(self):
        punctuation_path = os.path.join(self.ROOT_DIR, "data", self._PUNCTUATION_FILE)
        if os.path.exists(punctuation_path):
            return [x.strip() for x in open(punctuation_path).readlines()]
        else:
            raise FileNotFoundError(f"The file {punctuation_path} does not exist.")

    @cached_property
    def PUNCTUATION_MAP(self):
        return json.load(open(os.path.join(self.ROOT_DIR, "data", "punctuation.json")))

    @cached_property
    def LANG_CODE_TO_INDEX(self):
        return {lang: i for i, lang in enumerate(self.LANGINFO.index)}

    @cached_property
    def SEPARATORS(self):
        return defaultdict(
            lambda: " ", {lang: ("" if row["no_whitespace"] else " ") for lang, row in self.LANGINFO.iterrows()}
        )


Constants = ConstantsClass()


@dataclass
class LabelArgs:
    newline_remove_prob: float = 1.0
    auxiliary_remove_prob: float = 0.5
    hyphen_remove_prob: float = 0.9
    newline_whitespace_prob: float = 0.99
    hyphen_smooth_prob: float = 0.9
    newline_chars: List[str] = field(default_factory=lambda: ["\n"])
    auxiliary_chars: List[str] = field(default_factory=lambda: [])
    hyphen_chars: List[str] = field(default_factory=lambda: ["-", "‐"])
    use_auxiliary: bool = False
    custom_punctuation_file: str = None
    retain_first_consecutive_punctuation: bool = True
    non_whitespace_remove_spaces: bool = True
    case_corruption_prob_after_newline: float = 0.0
    case_corruption_prob_after_punct: float = 0.0
    corrupt_entire_chunk_prob: float = 0.0
    corrupt_entire_chunk_strategy: str = "mix"
    corrupt_entire_chunk_prob_full: float = 0.5
    use_all_labels: bool = False
    use_all_labels_max_length: int = 3

    def __post_init__(self):
        if self.custom_punctuation_file:
            Constants.set_punctuation_file(self.custom_punctuation_file)
        else:
            Constants.set_punctuation_file("punctuation.txt")
        self.auxiliary_chars = Constants.DEFAULT_PUNCTUATION_FILE


def get_label_dict(label_args):
    label_dict = {}

    for i, c in enumerate(Constants.PUNCTUATION_CHARS):
        label_dict[ord(c)] = 1 + Constants.AUX_OFFSET + i

    for c in label_args.newline_chars:
        label_dict[ord(c)] = 1 + Constants.NEWLINE_INDEX

    return label_dict


def get_subword_label_dict(label_args, tokenizer):
    label_dict = {}

    n_unks = 0
    # Map auxiliary characters to token IDs with labels
    logger.info(f"Using {Constants.PUNCTUATION_CHARS} auxiliary characters.")
    for i, c in enumerate(Constants.PUNCTUATION_CHARS):
        token_id = tokenizer.convert_tokens_to_ids(c)
        label_dict[token_id] = 1 + Constants.AUX_OFFSET + i
        logger.warning(
            f"auxiliary character {c} has token ID {token_id} and label {label_dict[token_id]}, decoded: {tokenizer.decode([token_id])}"
        )
        if token_id == tokenizer.unk_token_id:
            n_unks += 1
        if label_args.use_all_labels:
            # check where c is in tokenizer.vocab keys
            for i_t, (token, token_idx) in enumerate(tokenizer.vocab.items()):
                if (
                    c in token
                    and token_idx not in label_dict
                    and len(token) < label_args.use_all_labels_max_length
                    and not any(i.isdigit() for i in token)
                ):
                    label_dict[token_idx] = 1 + Constants.AUX_OFFSET + i
                    logger.warning(
                        f"Special auxiliary character {c} has token ID {token_idx} and label {label_dict[token_idx]}, decoded: {tokenizer.decode([token_idx])}"
                    )

    logger.info(f"found {n_unks} UNK tokens in auxiliary characters")

    # Map newline characters to token IDs with labels
    for c in label_args.newline_chars:
        token_id = tokenizer.convert_tokens_to_ids(c)
        label_dict[token_id] = 1 + Constants.NEWLINE_INDEX
        logger.warning(f"newline character {c} has token ID {token_id} and label {label_dict[token_id]}, decoded:")
        logger.warning(f"{tokenizer.decode([token_id])}")

    return label_dict


# numerically more stable sigmoid taken from
# https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    # See comment to the answer when it comes to dtype
    result = np.empty_like(x, dtype=np.float)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result


def encode(text):
    return [ord(c) for c in text]


def hash_encode(encoding, num_hashes=8, num_buckets=8192):
    if num_hashes > len(PRIMES):
        raise ValueError(f"`num_hashes` must be <= {len(PRIMES)}")

    hash_ids = np.zeros((len(encoding), num_hashes), dtype=np.int64)
    for i in range(num_hashes):
        shard_ids = (encoding + 1) * PRIMES[i]
        hash_ids[:, i] = shard_ids % num_buckets

    return hash_ids


def label(input_ids, label_dict):
    return [label_dict.get(input_id, 0) for input_id in input_ids[1:]] + [0]


def lang_code_to_lang(lang_code):
    from iso639 import languages

    if lang_code == "el":
        return "Greek"
    elif lang_code == "tl":
        return "Filipino"
    elif lang_code == "ug":
        return "Uyghur"

    try:
        return languages.get(alpha2=lang_code).name
    except KeyError:
        return languages.get(part3=lang_code).name


def corrupt(text: str, do_lowercase: bool, do_remove_punct: bool):
    if do_lowercase:
        text = text.lower()
    if do_remove_punct:
        for punct in Constants.PUNCTUATION_CHARS:
            text = text.replace(punct, "")
    return text


def corrupt_asr(text: str, lang):
    if text is None:
        return None

    separator = Constants.SEPARATORS.get(lang, " ")

    sentences = text.split("\n")

    if separator == "":
        corrupted_sentences = [
            "".join([char for char in sentence if char not in Constants.PUNCTUATION_CHARS]).lower()
            for sentence in sentences
        ]
        return corrupted_sentences

    try:
        tokenizer = MosesTokenizer(lang)

        tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
        corrupted_tokenized_sentences = [
            [token for token in tokens if token not in Constants.PUNCTUATION_CHARS] for tokens in tokenized_sentences
        ]
        corrupted_sentences = [
            tokenizer.detokenize(corrupted_tokens).lower() for corrupted_tokens in corrupted_tokenized_sentences
        ]
    except:
        corrupted_sentences = [
            "".join([char for char in sentence if char not in Constants.PUNCTUATION_CHARS]).lower()
            for sentence in sentences
        ]
        return corrupted_sentences

    return corrupted_sentences


# does the steps in Figure 2 of the paper
def corrupt_training(
    input_ids,
    block_ids,
    lang,
    label_args,
    label_dict,
    pack_samples=False,
    min_length=None,
    tokenizer=None,
):
    input_ids = input_ids.copy()
    block_ids = block_ids.copy()
    if random.random() < label_args.corrupt_entire_chunk_prob:
        # choose corruption strategy
        if label_args.corrupt_entire_chunk_strategy == "mix":
            corrupt_strategy = "full" if random.random() < label_args.corrupt_entire_chunk_prob_full else "asr"
        else:
            corrupt_strategy = label_args.corrupt_entire_chunk_strategy

        input_text = tokenizer.decode(input_ids)
        if corrupt_strategy == "tokenizer":
            if not tokenizer:
                raise NotImplementedError()
            corrupted = corrupt(input_text, do_lowercase=True, do_remove_punct=False)
            input_ids = tokenizer.encode(corrupted, add_special_tokens=False, verbose=False)
            # remove ALL punct *tokens*
            auxiliary_remove_prob = 1.0
        elif corrupt_strategy == "full":
            # remove all punct *characters*
            corrupted = corrupt(input_text, do_lowercase=True, do_remove_punct=True)
            input_ids = tokenizer.encode(corrupted, add_special_tokens=False, verbose=False)
            auxiliary_remove_prob = 1.0  # just for safety/consistency
        elif corrupt_strategy == "asr":
            if not tokenizer:
                raise NotImplementedError()
            corrupted_sentences = corrupt_asr(input_text, lang)
            corrupted_text = "\n".join(corrupted_sentences)
            input_ids = tokenizer.encode(corrupted_text, add_special_tokens=False, verbose=False)
            auxiliary_remove_prob = 0.0  # do not remove additional characters.
        block_ids = [0] * len(input_ids)

    else:
        auxiliary_remove_prob = label_args.auxiliary_remove_prob

    labels = label(input_ids, label_dict)

    separator = Constants.SEPARATORS[lang]

    try:
        i = next(index for index, label in enumerate(labels) if label != 0)
    except StopIteration:
        return input_ids, block_ids, labels

    if tokenizer:
        # account for CLS and SEP token, added later
        min_length = min_length - 2 if min_length is not None else None
    while min_length is None or len(input_ids) > min_length:
        if labels[i] == Constants.NEWLINE_INDEX + 1:
            if random.random() < label_args.newline_remove_prob:
                if separator == " " and random.random() < label_args.newline_whitespace_prob:
                    if tokenizer:
                        # inserting " " leaks \n information
                        # the token is never there naturally, so it is a 1:1 proxy for \n
                        del input_ids[i + 1]
                        del labels[i + 1]
                        del block_ids[i + 1]
                    else:
                        input_ids[i + 1] = ord(" ")
                else:
                    del input_ids[i + 1]
                    del labels[i + 1]

                    if pack_samples:
                        last_index_in_block = i
                        while (
                            last_index_in_block + 1 == len(block_ids)
                            or last_index_in_block < len(block_ids)
                            and block_ids[last_index_in_block + 1] == block_ids[last_index_in_block]
                        ):
                            last_index_in_block += 1
                        input_ids.insert(last_index_in_block, 0)
                        labels.insert(last_index_in_block, 0)
                    else:
                        del block_ids[i + 1]
                    if (
                        tokenizer
                        and separator == ""
                        and label_args.non_whitespace_remove_spaces
                        and i + 1 < len(input_ids)
                    ):
                        # tokenizer.decode() retains the space that leaks the information
                        # so we need to get the position within the tokenized text and then remove the space
                        # (so there is no more space when fed into the tokenizer call)
                        if input_ids[i + 1] == tokenizer.convert_tokens_to_ids("▁"):
                            # remove artificial space
                            del input_ids[i + 1]
                            del labels[i + 1]
                            del block_ids[i + 1]
                        if i + 1 < len(input_ids):
                            next_token = tokenizer.convert_ids_to_tokens(input_ids[i + 1])
                            if next_token.startswith("▁"):
                                # next token starts with _ --> remove the _ from the token and re-tokenize
                                remove_next = False
                                remaining_token = tokenizer.convert_ids_to_tokens(input_ids[i + 1])
                                if len(remaining_token) > 1:
                                    # ▁Test --> Test
                                    remaining_token = remaining_token[1:]
                                else:
                                    # ▁ --> remove
                                    remove_next = True
                                remaining_id = tokenizer.convert_tokens_to_ids(remaining_token)
                                # replace the token with the remaining token
                                if remaining_id != tokenizer.unk_token_id:
                                    input_ids[i + 1] = remaining_id
                                else:
                                    # UNK token, remove it
                                    remove_next = True
                                if remove_next:
                                    del input_ids[i + 1]
                                    del labels[i + 1]
                                    del block_ids[i + 1]
                if random.random() < label_args.case_corruption_prob_after_newline and i + 1 < len(input_ids):
                    input_ids, labels, block_ids = _corrupt_case(tokenizer, input_ids, labels, block_ids, i)

        elif label_args.use_auxiliary and labels[i] > Constants.AUX_OFFSET:  # auxiliary
            if pack_samples:
                raise NotImplementedError()

            if random.random() < auxiliary_remove_prob:
                removed_aux_char = False
                if label_args.retain_first_consecutive_punctuation:
                    # remove only if the next token is not a newline
                    # this retains the current auxiliary character, even though we decided to remove it
                    # it may skew the statistics since an auxiliary character is a better proxy for a newline
                    if labels[i + 1] != 1:
                        del input_ids[i + 1]
                        del labels[i + 1]
                        del block_ids[i + 1]
                        removed_aux_char = True
                else:
                    # in case of something like ".\n", this removes the "." and the \n label (=1)
                    # so the newline in the text is kept, but the label is removed!
                    del input_ids[i + 1]
                    del labels[i + 1]
                    del block_ids[i + 1]
                    removed_aux_char = True
                if (
                    random.random() < label_args.case_corruption_prob_after_punct
                    and removed_aux_char
                    and i + 1 < len(input_ids)
                ):
                    input_ids, labels, block_ids = _corrupt_case(tokenizer, input_ids, labels, block_ids, i)

        try:
            i = i + 1 + next(index for index, label in enumerate(labels[i + 1 :]) if label != 0)
        except StopIteration:
            break

    return input_ids, block_ids, labels


def _corrupt_case(tokenizer: AutoTokenizer, input_ids: List[int], labels: List[int], block_ids: List[int], i: int):
    if not tokenizer:
        raise NotImplementedError()

    token = tokenizer.convert_ids_to_tokens(input_ids[i + 1])
    insert_ = token.startswith("▁")
    if insert_:
        token = token[1:]

    do_exchange = token.istitle()
    if do_exchange:
        token = token.lower()

        # Re-tokenize
        token_ids = tokenizer.encode(token if not insert_ else "▁" + token, add_special_tokens=False)
        if len(token_ids) == 0 or input_ids[i + 1] == tokenizer.unk_token_id:
            # UNK or whitespace token, remove it
            del input_ids[i + 1]
            del labels[i + 1]
            del block_ids[i + 1]
        else:
            if token_ids[0] == tokenizer.convert_tokens_to_ids("▁"):
                token_ids = token_ids[1:]
            elif len(token_ids) > 1:
                # Replace the token with the remaining token
                input_ids[i + 1] = token_ids[0]
                for token_id in token_ids[1:]:
                    input_ids.insert(i + 2, token_id)
                    labels.insert(i + 2, 0)
                    block_ids.insert(i + 2, block_ids[i + 1])
            elif len(token_ids) == 1:
                input_ids[i + 1] = token_ids[0]

    return input_ids, labels, block_ids


def indices_to_sentences(text, indices, strip_whitespace=False):
    sentences = []

    offset = 0
    idx = 0
    for idx in indices:
        idx = idx + 1
        while idx < len(text) and text[idx].isspace():
            idx += 1

        sentence = text[offset:idx]
        if strip_whitespace:
            # NB: I would have thought that this is slower than
            # adjusting the start and end indices since there are
            # two string copies, but it seems to be faster
            # (at least on short strings). more reason to port to Rust?
            sentence = sentence.strip()

        if len(sentence) > 0:
            sentences.append(sentence)

        offset = idx

    if idx != len(text):
        last_sentence = text[idx:]
        if strip_whitespace:
            last_sentence = last_sentence.strip()

        if len(last_sentence) > 0:
            sentences.append(last_sentence)

    return sentences


def reconstruct_sentences(text, partial_sentences):
    # for consistency
    partial_sentences = [x.strip() for x in partial_sentences]

    fixed_sentences = []
    i = 0
    for x in partial_sentences:
        try:
            idx = i + text[i:].index(x[:-1])
        except ValueError:
            reduced = x[:-2]

            while len(reduced) > 0 and not reduced.isspace():
                idx = i + text[i:].find(reduced)

                if idx == -1:
                    reduced = reduced[:-1]
                else:
                    break

            if idx == -1:
                raise ValueError("irrecoverable error in reconstruction")

        if idx > i:
            fixed_sentences.append(text[i:idx])
            i = idx

    if i != len(text):
        fixed_sentences.append(text[i:])

    return fixed_sentences


if __name__ == "__main__":
    # test corrupt function
    from tokenizers import AddedToken

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
    text = "That's right, Five!\n!\n!!!\n!\n Always lay the!Blame on others!"
    input_ids = tokenizer(text)["input_ids"]
    block_ids = [0] * len(input_ids)
    label_args = LabelArgs(
        custom_punctuation_file="punctuation_xlmr_unk.txt",
        use_auxiliary=True,
        auxiliary_remove_prob=1.0,
        newline_whitespace_prob=1.0,
        case_corruption_prob_after_punct=1.0,
        case_corruption_prob_after_newline=1.0,
        use_all_labels=True,
    )
    label_dict = get_subword_label_dict(label_args, tokenizer)
    print(len(label_dict))
    # print all tokens with a number in it (from label_dict only)
    print(
        [tokenizer.decode(input_id) for input_id in input_ids if any(c.isdigit() for c in tokenizer.decode(input_id))]
    )
    # corrupt
    input_ids, block_ids, labels = corrupt_training(
        input_ids, block_ids, "en", label_args, label_dict, tokenizer=tokenizer
    )
    # print(input_ids)
    # print(labels)
    # print(tokenizer.tokenize(text))
    # print([(tokenizer.decode([input_id]), label) for input_id, label in zip(input_ids, labels)])
    # print("newline labels in text:")
    # print(np.where(np.array(labels) == 1))
    # print("newline ids in output text:")
    # print(np.where(np.array(input_ids) == tokenizer.all_special_ids[-1]))
    # print(tokenizer.decode(input_ids))
    # print(tokenizer.decode(input_ids))

    # ords = [ord(c) for c in text]
    # block_ords = [0] * len(ords)
    # label_args = LabelArgs(use_auxiliary=True, auxiliary_remove_prob=1.0)
    # label_dict = get_label_dict(label_args)

    # ords, block_ords, labels = corrupt(ords, block_ords, "en", label_args, label_dict)
    # print("ords", ords)
    # print("labels", labels)
    # print("newline labels in text:")
    # print(np.where(np.array(labels) == 1))
    # print("newline ids in output text:")
    # print(np.where(np.array([ord("\n")]) == ords))

import pandas


class Corpus:
    COLUMN_NAMES = ["source", "user_name", "sen_id", "sen_num", "token", "categ"]

    def __init__(self, source):
        """Initialize corpus with data from source.

        If source is an object of type Corpus, makes a copy of its data, otherwise assumes source is the name of a CSV
        file to read data from.
        """
        if isinstance(source, type(self)):
            self._data = source._data.copy()
        else:
            self._data = pandas.read_csv(source, header=0, names=self.COLUMN_NAMES, dtype=str, na_filter=False)

    def copy(self):
        """Returns deep copy of self."""
        return Corpus(self)

    def to_csv(self, file_name, *, linesep=None):
        """Write corpus to file file_name in CSV format.

        Uses the OS default as line separator, or linesep if provided (typically "\\n" for Unix, "\\r\\n" for Windows).
        """
        if linesep != None:
            self._data.to_csv(file_name, index=False, lineterminator=linesep)
        else:
            self._data.to_csv(file_name, index=False)

    def get_tokens(self, *, index=False, sort_posts=False):
        """Get a list of all tokens and corresponding tags.

        Returns a tuple of:
          - if index is True, a list of indices (of type int) which correspond to row numbers in the CSV file
          - a list of tokens (of type str)
          - a list of tags (of type str)
        Think of it as (row numbers plus) columns "token" and "categ" from the CSV file.
        By default, the lists are sorted by index, pass sort_posts=True to sort by post id (i.e. by "sen_id") instead.
        Indices can be used to update tags using set_tag() or set_tags().
        """
        if sort_posts:
            data = self._data.sort_values(by=["sen_id", "sen_num"])
        else:
            data = self._data

        idxs = list(data.index)
        toks = list(data["token"])
        tags = list(data["categ"])

        if index:
            return idxs, toks, tags
        else:
            return toks, tags

    def get_posts(self, *, index=False, sort_posts=False):
        """For each post, get a list of its tokens and corresponding tags.

        Returns a tuple of:
          - if index is True, a list of lists of indices (of type int) which correspond to row numbers in the CSV file
          - a list of lists of tokens (of type str)
          - a list of lists of tags (of type str)
        where each sublist represents one post (i.e. its elements share the same "sen_id").
        Think of it as (row numbers plus) columns "token" and "categ" from the CSV file, grouped by "sen_id".
        By default, the lists are sorted by index, pass sort_posts=True to sort by post id (i.e. by "sen_id") instead.
        Indices can be used to update tags using set_tag() or set_tags().
        """
        if sort_posts:
            data = self._data.sort_values(by=["sen_id", "sen_num"])
        else:
            data = self._data

        idxs, toks, tags = [], [], []

        sen_ids = data["sen_id"].drop_duplicates()
        for sen_id in sen_ids:
            post = data[data["sen_id"] == sen_id]
            idxs.append(list(post.index))
            toks.append(list(post["token"]))
            tags.append(list(post["categ"]))

        if index:
            return idxs, toks, tags
        else:
            return toks, tags

    def get_sentences(self, *, index=False, sort_posts=False):
        """For each sentence, get a list of its tokens and corresponding tags.

        Returns a tuple of:
          - if index is True, a list of lists of indices (of type int) which correspond to row numbers in the CSV file
          - a list of lists of tokens (of type str)
          - a list of lists of tags (of type str)
        where each sublist represents one sentence (i.e. its elements share the same "sen_id" and "sen_num").
        Think of it as (row numbers plus) columns "token" and "categ" from the CSV file, grouped by "sen_id"/"sen_num".
        By default, the lists are sorted by index, pass sort_posts=True to sort by post id (i.e. by "sen_id") instead.
        Indices can be used to update tags using set_tag() or set_tags().
        """
        if sort_posts:
            data = self._data.sort_values(by=["sen_id", "sen_num"])
        else:
            data = self._data

        idxs, toks, tags = [], [], []

        sen_ids = data["sen_id"].drop_duplicates()
        for sen_id in sen_ids:
            post = data[data["sen_id"] == sen_id]
            sen_nums = post["sen_num"].drop_duplicates()
            post_idxs, post_toks, post_tags = [], [], []
            for sen_num in sen_nums:
                sent = post[post["sen_num"] == sen_num]
                post_idxs.extend(list(sent.index))
                post_toks.extend(list(sent["token"]))
                post_tags.extend(list(sent["categ"]))

            idxs.append(post_idxs)
            toks.append(post_toks)
            tags.append(post_tags)

        if index:
            return idxs, toks, tags
        else:
            return toks, tags

    def set_tag(self, idx, tag):
        """Update a single tag.

        The entry at row number idx/column "categ" is set to value tag.
        The correct row number may be obtained by passing index=True to get_tokens(), get_posts() or get_sentences().
        """
        self._data.at[idx, "categ"] = tag

    def set_tags(self, idxs, tags):
        """Update multiple tags.

        For each row number in idxs, the entry in column "categ" is set to the corresponding value in tags.
        The correct row numbers may be obtained by passing index=True to get_tokens(), get_posts() or get_sentences().
        """
        for idx, tag in zip(idxs, tags):
            self.set_tag(idx, tag)
