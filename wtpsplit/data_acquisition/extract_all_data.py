import glob
import gzip
import json
import os
import random
from dataclasses import dataclass
from io import BytesIO

import conllu
import numpy as np
import pandas as pd
import requests
import torch
from datasets import load_dataset
from mosestokenizer import MosesTokenizer
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from utils.utils import Corpus
from wtpsplit.evaluation import preprocess_sentence
from wtpsplit.utils import Constants

UD_TREEBANK_PATH = "../data/ud-treebanks-v2.13"  # source: https://universaldependencies.org/#download

ERSATZ_DATA_PATH = "../data/ersatz-test-suite/segmented"  # source: https://github.com/rewicks/ersatz-test-suite

ERSATZ_TEST_DATASETS = {
    "ar": "iwsltt2017.ar",
    "cs": "wmt20.cs-en.cs",
    "de": "wmt20.de-en.de",
    "en": "wsj.03-06.en",
    "es": "wmt13.es-en.es",
    "et": "wmt18.et-en.et",
    "fi": "wmt19.fi-en.fi",
    "fr": "wmt20.fr-de.fr",
    "gu": "wmt19.gu-en.gu",
    "hi": "wmt14.hi-en.hi",
    "iu": "wmt20.iu-en.iu",
    "ja": "wmt20.ja-en.ja",
    "kk": "wmt19.kk-en.kk",
    "km": "wmt20.km-en.km",
    "lt": "wmt19.lt-en.lt",
    "lv": "wmt17.lv-en.lv",
    "pl": "wmt20.pl-en.pl",
    "ps": "wmt20.ps-en.ps",
    "ro": "wmt16.ro-en.ro",
    "ru": "wmt20.ru-en.ru",
    "ta": "wmt20.ta-en.ta",
    "tr": "wmt18.tr-en.tr",
    "zh": "wmt20.zh-en.zh",
}
ERSATZ_TRAIN_DATASETS = {
    "ar": "news-commentary-v15.dev.ar",
    "cs": "wmt18.cs-en.cs",
    "de": "wmt19.de-en.de",
    "en": "merged.nc-wsj.en",
    "es": None,
    "et": "newscrawl.2019.dev.et",
    "fi": "wmt18.fi-en.fi",
    "fr": "wmt15.fr-en.fr",
    "gu": "newscrawl.2019.dev.gu",
    "hi": "newscrawl.2013.dev.hi",
    "iu": "nhi-3.0.iu",
    "ja": "newscrawl.2019.dev.ja",
    "kk": "newscrawl.2019.dev.kk",
    "km": "wikimatrix.dev.km",
    "lt": "newscrawl.2019.dev.lt",
    "lv": "newscrawl.2019.dev.lv",
    "pl": "newscrawl.2019.dev.pl",
    "ps": "wikimatrix.dev.ps",
    "ro": "newscrawl.2019.dev.ro",
    "ru": "wmt18.ru-en.ru",
    "ta": "newscrawl.2019.dev.ta",
    "tr": "wmt16.tr-en.tr",
    "zh": "wmt18.zh-en.zh",
}


punct_chars = set(Constants.PUNCTUATION_CHARS)


def corrupt_asr(sentences, lang):

    if sentences is None:
        return None

    separator = Constants.SEPARATORS.get(lang, " ")

    if separator == "":
        corrupted_sentences = [
            preprocess_sentence("".join([char for char in sentence if char not in punct_chars]).lower())
            for sentence in sentences
        ]
        return corrupted_sentences

    try:
        tokenizer = MosesTokenizer(lang)
    except:
        corrupted_sentences = [
            preprocess_sentence("".join([char for char in sentence if char not in punct_chars]).lower())
            for sentence in sentences
        ]
        return corrupted_sentences

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    corrupted_tokenized_sentences = [
        [token for token in tokens if token not in punct_chars] for tokens in tokenized_sentences
    ]

    corrupted_sentences = [
        preprocess_sentence(tokenizer.detokenize(corrupted_tokens).lower())
        for corrupted_tokens in corrupted_tokenized_sentences
    ]

    return corrupted_sentences


def corrupt_social_media(sentences, lang):

    if sentences is None:
        return None

    corrupted_sentences = []
    for sentence in sentences:
        if random.random() < 0.5:
            sentence = "".join([char for char in sentence if char not in punct_chars])
        if random.random() < 0.5:
            sentence = sentence.lower()

        for punct in punct_chars:
            count = 0
            while random.random() < 0.5:
                count += 1
            sentence = sentence.replace(punct, punct * count)

        sentence = preprocess_sentence(sentence)
        corrupted_sentences.append(sentence)

    return corrupted_sentences


@dataclass
class Args:
    output_file: str = "../data/preprocessed_training_data/all_data_11_05.pth"
    include_train_data: bool = True
    cache_dir: str = "../data/cache/"


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    eval_data = {lang_code: {"sentence": {}, "compound": {}} for lang_code in Constants.LANGINFO.index}

    # Ersatz data
    for lang_code in tqdm(Constants.LANGINFO.index):
        if lang_code in ERSATZ_TEST_DATASETS:
            eval_data[lang_code]["sentence"]["ersatz"] = {
                "meta": {
                    "train_data": (
                        [
                            preprocess_sentence(line)
                            for line in open(
                                os.path.join(
                                    ERSATZ_DATA_PATH,
                                    lang_code,
                                    ERSATZ_TRAIN_DATASETS[lang_code],
                                )
                            )
                        ]
                        if args.include_train_data and ERSATZ_TRAIN_DATASETS[lang_code] is not None
                        else None
                    ),
                },
                "data": [
                    preprocess_sentence(line)
                    for line in open(os.path.join(ERSATZ_DATA_PATH, lang_code, ERSATZ_TEST_DATASETS[lang_code]))
                ],
            }

            eval_data[lang_code]["sentence"]["ersatz-corrupted-asr"] = {
                "meta": {
                    "train_data": (
                        corrupt_asr(
                            (
                                eval_data[lang_code]["sentence"]["ersatz"]["meta"]["train_data"][:10000]
                                if eval_data[lang_code]["sentence"]["ersatz"]["meta"]["train_data"] is not None
                                else None
                            ),
                            lang_code,
                        )
                    )
                },
                "data": corrupt_asr(
                    eval_data[lang_code]["sentence"]["ersatz"]["data"][:10000],
                    lang_code,
                ),
            }

            eval_data[lang_code]["sentence"]["ersatz-corrupted-social-media"] = {
                "meta": {
                    "train_data": (
                        corrupt_social_media(
                            (
                                eval_data[lang_code]["sentence"]["ersatz"]["meta"]["train_data"][:10000]
                                if eval_data[lang_code]["sentence"]["ersatz"]["meta"]["train_data"] is not None
                                else None
                            ),
                            lang_code,
                        )
                    )
                },
                "data": corrupt_social_media(
                    eval_data[lang_code]["sentence"]["ersatz"]["data"][:10000],
                    lang_code,
                ),
            }

    # UD + OPUS100 sentences + TED + NLLB
    for lang_code in tqdm(Constants.LANGINFO.index):
        opus_dset_name = Constants.LANGINFO.loc[lang_code, "opus100"]

        if opus_dset_name not in (np.nan, None):
            other_lang_code = set(opus_dset_name.split("-")) - {lang_code}
            assert len(other_lang_code) == 1
            other_lang_code = other_lang_code.pop()

            dset_args = ["opus100", opus_dset_name]

            try:
                opus100_sentences = [
                    preprocess_sentence(sample[lang_code])
                    for sample in load_dataset(*dset_args, split="test", cache_dir=args.cache_dir)["translation"]
                    if sample[lang_code].strip() != sample[other_lang_code].strip()
                ]
                try:
                    opus100_train_sentences = [
                        preprocess_sentence(sample[lang_code])
                        for sample in load_dataset(*dset_args, split="train", cache_dir=args.cache_dir)["translation"]
                        if sample[lang_code].strip() != sample[other_lang_code].strip()
                    ]
                except ValueError:
                    opus100_train_sentences = None
            except ValueError:
                opus100_sentences = [
                    preprocess_sentence(sample[lang_code])
                    for sample in load_dataset(*dset_args, split="train", cache_dir=args.cache_dir)["translation"]
                    if sample[lang_code].strip() != sample[other_lang_code].strip()
                ]
                opus100_train_sentences = None

            opus100_train_sentences = opus100_train_sentences[:10000] if opus100_train_sentences is not None else None

            eval_data[lang_code]["sentence"]["opus100"] = {
                "meta": {"train_data": (opus100_train_sentences if args.include_train_data else None)},
                "data": opus100_sentences,
            }

            eval_data[lang_code]["sentence"]["opus100-corrupted-asr"] = {
                "meta": {
                    "train_data": (
                        corrupt_asr(
                            (opus100_train_sentences),
                            lang_code,
                        )
                        if args.include_train_data
                        else None
                    )
                },
                "data": corrupt_asr(opus100_sentences[:10000], lang_code),
            }

            eval_data[lang_code]["sentence"]["opus100-corrupted-social-media"] = {
                "meta": {
                    "train_data": (
                        corrupt_social_media(
                            (opus100_train_sentences),
                            lang_code,
                        )
                        if args.include_train_data
                        else None
                    )
                },
                "data": corrupt_social_media(opus100_sentences[:10000], lang_code),
            }

        if Constants.LANGINFO.loc[lang_code, "ud"] not in (np.nan, None):
            ud_data = conllu.parse(
                open(
                    glob.glob(
                        os.path.join(
                            UD_TREEBANK_PATH,
                            Constants.LANGINFO.loc[lang_code, "ud"],
                            "*-ud-test.conllu",
                        )
                    )[0]
                ).read()
            )

            try:
                ud_train_data = conllu.parse(
                    open(
                        glob.glob(
                            os.path.join(
                                UD_TREEBANK_PATH,
                                Constants.LANGINFO.loc[lang_code, "ud"],
                                "*-ud-train.conllu",
                            )
                        )[0]
                    ).read()
                )
                ud_train_sentences = [preprocess_sentence(sentence.metadata["text"]) for sentence in ud_train_data][
                    :10000
                ]
            except IndexError:
                ud_train_sentences = None

            ud_sentences = [preprocess_sentence(sentence.metadata["text"]) for sentence in ud_data]

            eval_data[lang_code]["sentence"]["ud"] = {
                "meta": {"train_data": (ud_train_sentences if args.include_train_data else None)},
                "data": ud_sentences,
            }

            eval_data[lang_code]["sentence"]["ud-corrupted-asr"] = {
                "meta": {
                    "train_data": (corrupt_asr(ud_train_sentences, lang_code) if args.include_train_data else None)
                },
                "data": corrupt_asr(ud_sentences, lang_code),
            }

            eval_data[lang_code]["sentence"]["ud-corrupted-social-media"] = {
                "meta": {
                    "train_data": (
                        corrupt_social_media(ud_train_sentences, lang_code) if args.include_train_data else None
                    )
                },
                "data": corrupt_social_media(ud_sentences, lang_code),
            }

        # TED 2020
        url = f"https://object.pouta.csc.fi/OPUS-TED2020/v1/mono/{lang_code}.txt.gz"
        res = requests.get(url)

        if res.status_code == 200:
            with gzip.open(BytesIO(res.content), "rt", encoding="utf-8") as f:
                sentences = f.read().splitlines()

            sentences = sentences[:20000]

            sentences = [preprocess_sentence(sentence) for sentence in sentences]

            train_sentences = sentences[: len(sentences) // 2]
            test_sentences = sentences[len(sentences) // 2 :]

            eval_data[lang_code]["sentence"]["ted2020-corrupted-asr"] = {
                "meta": {"train_data": (corrupt_asr(train_sentences, lang_code) if args.include_train_data else None)},
                "data": corrupt_asr(test_sentences, lang_code),
            }

            eval_data[lang_code]["sentence"]["ted2020-corrupted-social-media"] = {
                "meta": {
                    "train_data": (
                        corrupt_social_media(train_sentences, lang_code) if args.include_train_data else None
                    )
                },
                "data": corrupt_social_media(test_sentences, lang_code),
            }

        else:

            print(f"Failed to download TED2020 data for {lang_code}")

    for lang_code in ["ceb", "jv", "mn", "yo"]:
        url = f"https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/{lang_code}.txt.gz"
        res = requests.get(url)

        if res.status_code == 200:
            with gzip.open(BytesIO(res.content), "rt", encoding="utf-8") as f:
                sentences = f.read().splitlines()

            random.shuffle(sentences)  # because they come alphabetically sorted

            sentences = sentences[:20000]

            sentences = [preprocess_sentence(sentence) for sentence in sentences]

        else:
            raise Exception

        train_sentences = sentences[: len(sentences) // 2]
        test_sentences = sentences[len(sentences) // 2 :]

        eval_data[lang_code]["sentence"]["nllb"] = {
            "meta": {"train_data": (train_sentences if args.include_train_data else None)},
            "data": test_sentences,
        }

        eval_data[lang_code]["sentence"]["nllb-corrupted-asr"] = {
            "meta": {"train_data": (corrupt_asr(train_sentences, lang_code) if args.include_train_data else None)},
            "data": corrupt_asr(test_sentences, lang_code),
        }

        eval_data[lang_code]["sentence"]["nllb-corrupted-social-media"] = {
            "meta": {
                "train_data": (corrupt_social_media(train_sentences, lang_code) if args.include_train_data else None)
            },
            "data": corrupt_social_media(test_sentences, lang_code),
        }

    # UD Code-Switching Corpora

    # UD_Turkish_German-SAGT

    ud_data = conllu.parse(
        open(
            glob.glob(
                os.path.join(
                    UD_TREEBANK_PATH,
                    "UD_Turkish_German-SAGT",
                    "*-ud-test.conllu",
                )
            )[0]
        ).read()
    )

    ud_train_data = conllu.parse(
        open(
            glob.glob(
                os.path.join(
                    UD_TREEBANK_PATH,
                    "UD_Turkish_German-SAGT",
                    "*-ud-train.conllu",
                )
            )[0]
        ).read()
    )

    ud_train_sentences = [preprocess_sentence(sentence.metadata["text"]) for sentence in ud_train_data]
    ud_test_sentences = [preprocess_sentence(sentence.metadata["text"]) for sentence in ud_data]

    eval_data["tr-de"] = {}
    eval_data["tr-de"]["sentence"] = {}

    eval_data["tr-de"]["sentence"]["code-switching"] = {
        "meta": {"train_data": ud_train_sentences},
        "data": ud_test_sentences,
    }

    eval_data["tr-de"]["sentence"]["code-switching-corrupted-asr"] = {
        "meta": {"train_data": corrupt_asr(ud_train_sentences, "en")},
        "data": corrupt_asr(ud_test_sentences, "en"),
    }

    eval_data["tr-de"]["sentence"]["code-switching-corrupted-social-media"] = {
        "meta": {"train_data": corrupt_social_media(ud_train_sentences, "en")},
        "data": corrupt_social_media(ud_test_sentences, "en"),
    }

    # UD_Spanish_English-Miami

    ud_data = conllu.parse(
        open(
            glob.glob(
                os.path.join(
                    UD_TREEBANK_PATH,
                    "UD_Spanish_English-Miami",
                    "*-ud-test.conllu",
                )
            )[0]
        ).read()
    )

    ud_sentences = [preprocess_sentence(sentence.metadata["text"]) for sentence in ud_data]
    ud_train_sentences = ud_sentences[len(ud_sentences) // 2 :]
    ud_test_sentences = ud_sentences[: len(ud_sentences) // 2]

    eval_data["es-en"] = {}
    eval_data["es-en"]["sentence"] = {}

    eval_data["es-en"]["sentence"]["code-switching"] = {
        "meta": {"train_data": ud_train_sentences},
        "data": ud_test_sentences,
    }

    eval_data["es-en"]["sentence"]["code-switching-corrupted-asr"] = {
        "meta": {"train_data": corrupt_asr(ud_train_sentences, "es")},
        "data": corrupt_asr(ud_test_sentences, "es"),
    }

    eval_data["es-en"]["sentence"]["code-switching-corrupted-social-media"] = {
        "meta": {"train_data": corrupt_social_media(ud_train_sentences, "es")},
        "data": corrupt_social_media(ud_test_sentences, "es"),
    }

    # Vietnamese--English

    canvec_corpus = pd.read_excel("../data/vietnamese-english/CanVEC_CSW.xlsx", header=None)
    # sentences are the first columnn
    sentences = canvec_corpus.iloc[:, 0].tolist()

    sentences = [preprocess_sentence(sentence) for sentence in sentences]

    train_sentences = sentences[: len(sentences) // 2]
    test_sentences = sentences[len(sentences) // 2 :]

    eval_data["vi-en"] = {}
    eval_data["vi-en"]["sentence"] = {}

    eval_data["vi-en"]["sentence"]["code-switching"] = {
        "meta": {"train_data": train_sentences},
        "data": test_sentences,
    }

    eval_data["vi-en"]["sentence"]["code-switching-corrupted-asr"] = {
        "meta": {"train_data": corrupt_asr(train_sentences, "vi")},
        "data": corrupt_asr(test_sentences, "vi"),
    }

    eval_data["vi-en"]["sentence"]["code-switching-corrupted-social-media"] = {
        "meta": {"train_data": corrupt_social_media(train_sentences, "vi")},
        "data": corrupt_social_media(test_sentences, "vi"),
    }

    # Denglisch

    DENGLISCH_PATH = "../data/denglisch/Manu_corpus.csv"
    denglisch_corpus = Corpus(DENGLISCH_PATH)
    all_tokens, all_labels = denglisch_corpus.get_sentences()

    tokenizer = MosesTokenizer("de")

    denglisch_posts = []

    for post_tokens, post_labels in zip(all_tokens, all_labels):
        post_sentences = []
        sentence_tokens = []
        sentence_labels = []
        for token, label in zip(post_tokens, post_labels):
            if token and token != "$newline$":
                if token == "$quote$":
                    token = '"'
                sentence_tokens.append(token)
                sentence_labels.append(label)
            if label == "<EOS>":
                if sentence_tokens:
                    sentence_text = preprocess_sentence(tokenizer.detokenize(sentence_tokens))
                    post_sentences.append((sentence_text, sentence_labels))
                sentence_tokens = []
                sentence_labels = []

        if sentence_tokens:
            sentence_text = preprocess_sentence(tokenizer.detokenize(sentence_tokens))
            post_sentences.append((sentence_text, sentence_labels))

        denglisch_posts.append(post_sentences)

    random.shuffle(denglisch_posts)

    denglisch_test_posts = denglisch_posts[len(denglisch_posts) // 2 :]
    denglisch_train_posts = denglisch_posts[: len(denglisch_posts) // 2]

    # flatten from list of lists to list
    denglisch_test_sentences = [
        sentence for sublist in denglisch_test_posts for sentence, labels in sublist if "1" in labels and "2" in labels
    ]
    denglisch_train_sentences = [
        sentence for sublist in denglisch_train_posts for sentence, labels in sublist if "1" in labels and "2" in labels
    ]

    eval_data["en-de"] = {}
    eval_data["en-de"]["sentence"] = {}
    eval_data["en-de"]["sentence"]["code-switching"] = {
        "meta": {"train_data": denglisch_train_sentences},
        "data": denglisch_test_sentences,
    }

    eval_data["en-de"]["sentence"]["code-switching-corrupted-asr"] = {
        "meta": {"train_data": corrupt_asr(denglisch_train_sentences, "de")},
        "data": corrupt_asr(denglisch_test_sentences, "de"),
    }

    eval_data["en-de"]["sentence"]["code-switching-corrupted-social-media"] = {
        "meta": {"train_data": corrupt_social_media(denglisch_train_sentences, "de")},
        "data": corrupt_social_media(denglisch_test_sentences, "de"),
    }

    denglisch_test_posts = [[sentence for sentence, labels in post] for post in denglisch_test_posts]
    denglisch_train_posts = [[sentence for sentence, labels in post] for post in denglisch_train_posts]

    eval_data["en-de"]["sentence"]["short-sequences"] = {
        "meta": {"train_data": denglisch_train_posts},
        "data": denglisch_test_posts,
    }

    eval_data["en-de"]["sentence"]["short-sequences-corrupted-asr"] = {
        "meta": {"train_data": [corrupt_asr(s, "de") for s in denglisch_train_posts]},
        "data": [corrupt_asr(s, "de") for s in denglisch_test_posts],
    }

    eval_data["en-de"]["sentence"]["short-sequences-corrupted-social-media"] = {
        "meta": {"train_data": [corrupt_social_media(s, "de") for s in denglisch_train_posts]},
        "data": [corrupt_social_media(s, "de") for s in denglisch_test_posts],
    }

    # Short sequences

    # serbian

    serbian_train_data = conllu.parse(open("../data/short-sequences/serbian/reldi-normtagner-sr-train.conllu").read())

    serbian_train_tweets = []
    tweet_sentences = []
    for sentence in serbian_train_data:
        if "newdoc id" in sentence.metadata:
            if tweet_sentences:
                serbian_train_tweets.append(tweet_sentences)
            tweet_sentences = []
        tweet_sentences.append(preprocess_sentence(sentence.metadata["text"]))

    if tweet_sentences:
        serbian_train_tweets.append(tweet_sentences)

    serbian_test_data = conllu.parse(open("../data/short-sequences/serbian/reldi-normtagner-sr-test.conllu").read())

    serbian_test_tweets = []
    tweet_sentences = []
    for sentence in serbian_test_data:
        if "newdoc id" in sentence.metadata:
            if tweet_sentences:
                serbian_test_tweets.append(tweet_sentences)
            tweet_sentences = []
        tweet_sentences.append(preprocess_sentence(sentence.metadata["text"]))

    if tweet_sentences:
        serbian_test_tweets.append(tweet_sentences)

    serbian_train_tweets = [tweet for tweet in serbian_train_tweets if len(tweet) > 1]
    serbian_test_tweets = [tweet for tweet in serbian_test_tweets if len(tweet) > 1]

    eval_data["sr"]["sentence"]["short-sequences"] = {
        "meta": {"train_data": serbian_train_tweets},
        "data": serbian_test_tweets,
    }

    eval_data["sr"]["sentence"]["short-sequences-corrupted-asr"] = {
        "meta": {"train_data": [corrupt_asr(s, "sr") for s in serbian_train_tweets]},
        "data": [corrupt_asr(s, "sr") for s in serbian_test_tweets],
    }

    eval_data["sr"]["sentence"]["short-sequences-corrupted-social-media"] = {
        "meta": {"train_data": [corrupt_social_media(s, "sr") for s in serbian_train_tweets]},
        "data": [corrupt_social_media(s, "sr") for s in serbian_test_tweets],
    }

    # slovenian

    slovenian_data = conllu.parse(
        open("../data/short-sequences/slovenian/Janes-Tag.3.0.CoNLL-U/janes-rsdo.ud.connlu").read()
    )

    slovenian_tweets = []
    tweet_sentences = []
    for sentence in slovenian_data:
        if "newdoc id" in sentence.metadata:
            if tweet_sentences:
                slovenian_tweets.append(tweet_sentences)
            tweet_sentences = []
        tweet_sentences.append(preprocess_sentence(sentence.metadata["text"]))

    if tweet_sentences:
        slovenian_tweets.append(tweet_sentences)

    random.shuffle(slovenian_tweets)

    # keep only if more than one sentence in a tweet
    slovenian_tweets = [tweet for tweet in slovenian_tweets if len(tweet) > 1]

    slovenian_train_tweeets = slovenian_tweets[: len(slovenian_tweets) // 2]
    slovenian_test_tweets = slovenian_tweets[len(slovenian_tweets) // 2 :]

    eval_data["sl"]["sentence"]["short-sequences"] = {
        "meta": {"train_data": slovenian_train_tweeets},
        "data": slovenian_test_tweets,
    }

    eval_data["sl"]["sentence"]["short-sequences-corrupted-asr"] = {
        "meta": {"train_data": [corrupt_asr(s, "sl") for s in slovenian_train_tweeets]},
        "data": [corrupt_asr(s, "sl") for s in slovenian_test_tweets],
    }

    eval_data["sl"]["sentence"]["short-sequences-corrupted-social-media"] = {
        "meta": {"train_data": [corrupt_social_media(s, "sl") for s in slovenian_train_tweeets]},
        "data": [corrupt_social_media(s, "sl") for s in slovenian_test_tweets],
    }

    # estonian

    estonian_data = conllu.parse(
        open("../data/short-sequences/estonian/EWTB_sentence_seg/EWTB_sent_token_manual.conllu").read()
    )

    estonian_paragraphs = []
    paragraph_sentences = []

    for sentence in estonian_data:

        if "newpar id" in sentence.metadata:
            if paragraph_sentences:
                estonian_paragraphs.append(paragraph_sentences)
            paragraph_sentences = []
        paragraph_sentences.append(preprocess_sentence(sentence.metadata["text"]))

    if paragraph_sentences:
        estonian_paragraphs.append(paragraph_sentences)

    # keep only if more than one sentence in a paragraph
    estonian_paragraphs = [paragraph for paragraph in estonian_paragraphs if len(paragraph) > 1]

    random.shuffle(estonian_paragraphs)

    estonian_train_paragraphs = estonian_paragraphs[: len(estonian_paragraphs) // 2]
    estonian_test_paragraphs = estonian_paragraphs[len(estonian_paragraphs) // 2 :]

    eval_data["et"]["sentence"]["short-sequences"] = {
        "meta": {"train_data": estonian_train_paragraphs},
        "data": estonian_test_paragraphs,
    }

    eval_data["et"]["sentence"]["short-sequences-corrupted-asr"] = {
        "meta": {"train_data": [corrupt_asr(s, "et") for s in estonian_train_paragraphs]},
        "data": [corrupt_asr(s, "et") for s in estonian_test_paragraphs],
    }

    eval_data["et"]["sentence"]["short-sequences-corrupted-social-media"] = {
        "meta": {"train_data": [corrupt_social_media(s, "et") for s in estonian_train_paragraphs]},
        "data": [corrupt_social_media(s, "et") for s in estonian_test_paragraphs],
    }

    langs = ["de", "en", "es", "fr", "it", "pt"]

    all_subset_data = {
        lang: {
            "laws": {"train": [], "test": []},
            "judgements": {"train": [], "test": []},
        }
        for lang in langs
    }

    for lang in tqdm(langs, desc="Legal data"):
        data_dir = f"../data/MultiLegalSBD/data/{lang}/gold/"

        if lang == "pt":
            all_files = glob.glob(f"{data_dir}/*.jsonl")
            subsets = [file.split("/")[-1].split(".jsonl")[0] for file in all_files]
        else:
            all_files = glob.glob(f"{data_dir}/*_test.jsonl")
            subsets = [file.split("/")[-1].split("_test.jsonl")[0] for file in all_files]

        for subset in subsets:

            if subset == "Constitution":
                continue

            if lang == "pt":
                train_data = [None]
            else:

                train_data = []

                with open(
                    f"../data/MultiLegalSBD/data/{lang}/gold/{subset}_train.jsonl",
                    "r",
                    encoding="utf-8",
                ) as f:
                    for line in f:
                        train_data.append(json.loads(line))

                train_subset_sentences = []
                for doc in train_data:
                    doc_sentences = []
                    text = doc["text"]
                    for span in doc["spans"]:
                        sentence = text[span["start"] : span["end"]]
                        doc_sentences.append(preprocess_sentence(sentence))
                    train_subset_sentences.append(doc_sentences)

            test_data = []

            if lang == "pt":
                test_data_file = f"../data/MultiLegalSBD/data/{lang}/gold/{subset}.jsonl"
            else:
                test_data_file = f"../data/MultiLegalSBD/data/{lang}/gold/{subset}_test.jsonl"

            with open(
                test_data_file,
                "r",
                encoding="utf-8",
            ) as f:
                for line in f:
                    test_data.append(json.loads(line))

            test_subset_sentences = []
            for doc in test_data:
                doc_sentences = []
                text = doc["text"]
                for span in doc["spans"]:
                    sentence = text[span["start"] : span["end"]]
                    doc_sentences.append(preprocess_sentence(sentence))
                test_subset_sentences.append(doc_sentences)

            eval_data[lang]["sentence"][f"legal-{subset}"] = {
                "meta": {"train_data": train_subset_sentences},
                "data": test_subset_sentences,
            }

            eval_data[lang]["sentence"][f"legal-{subset}-corrupted-asr"] = {
                "meta": {"train_data": [corrupt_asr(s, lang) for s in train_subset_sentences]},
                "data": [corrupt_asr(s, lang) for s in test_subset_sentences],
            }

            eval_data[lang]["sentence"][f"legal-{subset}-corrupted-social-media"] = {
                "meta": {"train_data": [corrupt_social_media(s, lang) for s in train_subset_sentences]},
                "data": [corrupt_social_media(s, lang) for s in test_subset_sentences],
            }

            subsets2set = {
                "CD_jug": "judgements",
                "gesCode": "laws",
                "CD_multi_legal": "judgements",
                "CD_wipolex": "judgements",
                "CivilCode": "laws",
                "CriminalCode": "laws",
                "CD_swiss_judgement": "judgements",
            }

            if lang != "en":
                set = subsets2set[subset]
            else:
                set = "judgements"

            all_subset_data[lang][set]["train"].extend(train_subset_sentences)
            all_subset_data[lang][set]["test"].extend(test_subset_sentences)

        # constitution

        if lang in ["de", "en"]:
            continue

        test_data = []
        test_data_file = f"../data/MultiLegalSBD/data/{lang}/gold/Constitution.jsonl"
        with open(
            test_data_file,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                test_data.append(json.loads(line))

        test_subset_sentences = []
        for doc in test_data:
            doc_sentences = []
            text = doc["text"]
            for span in doc["spans"]:
                sentence = text[span["start"] : span["end"]]
                doc_sentences.append(preprocess_sentence(sentence))
            test_subset_sentences.append(doc_sentences)

        eval_data[lang]["sentence"][f"legal-constitution"] = {
            "meta": {"train_data": None},
            "data": test_subset_sentences,
        }

        eval_data[lang]["sentence"][f"legal-constitution-corrupted-asr"] = {
            "meta": {"train_data": None},
            "data": [corrupt_asr(s, lang) for s in test_subset_sentences],
        }

        eval_data[lang]["sentence"][f"legal-constitution-corrupted-social-media"] = {
            "meta": {"train_data": None},
            "data": [corrupt_social_media(s, lang) for s in test_subset_sentences],
        }

        all_subset_data[lang]["laws"]["test"].extend(test_subset_sentences)

    for lang in all_subset_data:
        for set in ["laws", "judgements"]:
            eval_data[lang]["sentence"][f"legal-all-{set}"] = {
                "meta": {"train_data": all_subset_data[lang][set]["train"]},
                "data": all_subset_data[lang][set]["test"],
            }

            eval_data[lang]["sentence"][f"legal-all-{set}-corrupted-asr"] = {
                "meta": {"train_data": [corrupt_asr(s, lang) for s in all_subset_data[lang][set]["train"]]},
                "data": [corrupt_asr(s, lang) for s in all_subset_data[lang][set]["test"]],
            }

            eval_data[lang]["sentence"][f"legal-all-{set}-corrupted-social-media"] = {
                "meta": {"train_data": [corrupt_social_media(s, lang) for s in all_subset_data[lang][set]["train"]]},
                "data": [corrupt_social_media(s, lang) for s in all_subset_data[lang][set]["test"]],
            }

    torch.save(eval_data, args.output_file.replace(".pth", "-all.pth"))

    eval_data = {}
    eval_data["en"] = {}
    eval_data["en"]["sentence"] = {}

    lyric_data = torch.load("../data/mldbW_verses.pth")

    for lang in lyric_data.keys():
        for genre in lyric_data[lang]["sentence"].keys():
            train_data = lyric_data[lang]["sentence"][genre]["meta"]["train_data"]
            train_data = [[verse for verse in song if verse != ""] for song in train_data]
            test_data = lyric_data[lang]["sentence"][genre]["data"]
            test_data = [[verse for verse in song if verse != ""] for song in test_data]

            eval_data[lang]["sentence"]["lyrics-" + genre] = {
                "meta": {"train_data": train_data},
                "data": test_data,
            }

            eval_data[lang]["sentence"]["lyrics-" + genre + "-corrupted-asr"] = {
                "meta": {"train_data": [corrupt_asr(s, lang) for s in train_data]},
                "data": [corrupt_asr(s, lang) for s in test_data],
            }

            eval_data[lang]["sentence"]["lyrics-" + genre + "corrupted-social-media"] = {
                "meta": {"train_data": [corrupt_social_media(s, lang) for s in train_data]},
                "data": [corrupt_social_media(s, lang) for s in test_data],
            }

    torch.save(eval_data, args.output_file.replace(".pth", "-lyrics.pth"))
