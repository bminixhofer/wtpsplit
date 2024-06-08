import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
import pickle

ALL_DIR = Path("../data/permutation-test-data/")

raw_data = defaultdict(lambda: defaultdict(dict))
val_results = defaultdict(lambda: defaultdict(dict))


DATA_DIR = ALL_DIR / "all-stat-test-data/main_all"
LATEX_DIR = ALL_DIR / "p-values"
RESULTS_DATA_DIR = ALL_DIR / "results_data"

spacy_langs = open("../data/permutation-test-data/all-stat-test-data/spacy_m_langs.txt").read().splitlines()


for file in tqdm(DATA_DIR.glob("*IDX.json"), desc="Loading indices"):

    model = str(file.stem)[:-4]
    with open(file, "r") as f:
        data = json.load(f)
    for lang in data.keys():
        if file.stem.startswith("intrinsic_baselines_multi") and lang not in spacy_langs:
            continue

        for dataset in data[lang].keys():
            if (
                dataset.startswith("legal")
                or dataset.startswith("ted")
                or "corrupted-asr" in dataset
                or "short-sequences" in dataset
                or "code-switching" in dataset
            ):
                continue

            for model_type in data[lang][dataset].keys():

                if lang == "cs" and (model_type == "meta/meta-llama-3-8b-instruct" or model_type == "command-r"):
                    continue

                if model_type.startswith("spacy_sent"):
                    continue

                if (
                    (
                        model_type == "true_indices"
                        or model_type == "length"
                        or model_type == "lengths"
                        or model_type == "refused"
                    )
                    or data[lang][dataset][model_type] is None
                    or "predicted_indices" not in data[lang][dataset][model_type]
                ):
                    continue

                data_list = data[lang][dataset][model_type]["predicted_indices"]

                if data_list is None:
                    continue

                if len(data_list) == 0:
                    data_list = [[]]
                try:
                    if isinstance(data_list[0], int):
                        data_list = [data_list]
                except:
                    print(data_list)
                    print(lang, dataset, model_type)
                    raise Exception

                raw_data[lang][dataset][model + "-" + model_type] = data_list

                true_indices = data[lang][dataset][model_type]["true_indices"]

                if isinstance(true_indices[0], int):
                    true_indices = [true_indices]

                if "true_indicies" in raw_data[lang][dataset]:
                    assert raw_data[lang][dataset]["true_indices"] == true_indices
                else:
                    raw_data[lang][dataset]["true_indices"] = true_indices

                data_lengths = (
                    data[lang][dataset][model_type]["length"]
                    if "length" in data[lang][dataset][model_type]
                    else data[lang][dataset][model_type]["lengths"]
                )

                if isinstance(data_lengths, int):
                    data_lengths = [data_lengths]

                if "lengths" in raw_data[lang][dataset]:
                    assert (
                        raw_data[lang][dataset]["lengths"] == data_lengths
                    ), f"{lang}, {dataset}, {model_type}... [lengths assertion] before: {raw_data[lang][dataset]['lengths']} after: {data_lengths}"
                else:
                    raw_data[lang][dataset]["lengths"] = data_lengths


for file in tqdm(DATA_DIR.glob("*.json"), desc="Loading F1s"):
    if file.stem.endswith("IDX"):
        continue

    with open(file, "r") as f:
        data = json.load(f)

    model = file.stem

    for lang in data.keys():
        if file.stem.startswith("intrinsic_baselines_multi") and lang not in spacy_langs:
            continue

        for dataset in data[lang].keys():
            for model_type in data[lang][dataset].keys():
                if lang == "cs" and (model_type == "meta/meta-llama-3-8b-instruct" or model_type == "command-r"):
                    continue

                if model_type == "f1":
                    renamed_model_type = "llm"
                else:
                    renamed_model_type = model_type
                result = data[lang][dataset][model_type]

                if result is None or result == {}:
                    continue
                elif not isinstance(result, float):
                    result = result["f1"]

                val_results[lang][dataset][model + "-" + renamed_model_type] = result


all_systems_mapping = {
    "benjamin_wtp-canine-s-3l_b512_s64_u0.01_k10-punct": "WtP-P",
    "benjamin_wtp-canine-s-3l_b512_s64_u0.01_k10-t": "WtP-T",
    "benjamin_wtp-canine-s-3l_b512_s64_u0.01_k10-u": "WtP-U",
    "command-r_k10_s0Pv2-all-command-r": "C-R",
    "meta-llama-3-8b-instruct_k10_s0Pv2-all-meta/meta-llama-3-8b-instruct": "L-3",
    "xlmr-3l-v3_lc0.1-mix2-FT-33-33-33-v2-CorrSep_b512_s64_u0.25_k10-u": "SaT-SM",
    "xlmr-3l-v3_look48_lc0.1-mix2_b512_s64_u0.025_k10-t": "SaT-T",
    "xlmr-3l-v3_look48_lc0.1-mix2_b512_s64_u0.025_k10-u": "SaT-U",
    "xlmr-3l-v4_LL_lora-v2_ep30_s10k_b512_s64_u0.5_k10-t": "SaT-Lora-T",
    "xlmr-3l-v4_LL_lora-v2_ep30_s10k_b512_s64_u0.5_k10-u": "SaT-Lora-U",
    "intrinsic_baselines-spacy_dp": "spacy-dp",
    "intrinsic_baselines_multi-spacy_dp": "spacy-m",
}

main_table_exclude = [
    ["bn", "ud"],
    ["ceb", "ud"],
    ["es", "ersatz"],
    ["fr", "opus100"],
    ["hy", "opus100"],
    ["id", "ud"],
    ["jv", "ud"],
    ["mn", "opus100"],
    ["nl", "opus100"],
    ["ru", "opus100"],
    ["sq", "ud"],
    ["th", "ud"],
    ["yo", "opus100"],
    ["yo", "ud"],
]


for lang in raw_data.keys():
    for system in all_systems_mapping.keys():

        main_results = []
        if (
            "opus100" in raw_data[lang]
            and system in raw_data[lang]["opus100"]
            and [lang, "opus100"] not in main_table_exclude
        ):
            main_results.append(val_results[lang]["opus100"][system])
        if "ud" in raw_data[lang] and system in raw_data[lang]["ud"] and [lang, "ud"] not in main_table_exclude:
            main_results.append(val_results[lang]["ud"][system])
        if (
            "ersatz" in raw_data[lang]
            and system in raw_data[lang]["ersatz"]
            and [lang, "ersatz"] not in main_table_exclude
        ):
            main_results.append(val_results[lang]["ersatz"][system])

        if main_results == []:
            continue

        avg_f1 = sum(main_results) / len(main_results)

        preds_main_results_indicies = []
        trues_main_results_indicies = []
        lengths_main_results = []

        val_results[lang]["main_table_mean"][system] = avg_f1

        if (
            "opus100" in raw_data[lang]
            and system in raw_data[lang]["opus100"]
            and [lang, "opus100"] not in main_table_exclude
        ):
            preds_main_results_indicies.append(raw_data[lang]["opus100"][system][0])
            trues_main_results_indicies.append(raw_data[lang]["opus100"]["true_indices"])
            lengths_main_results.append(raw_data[lang]["opus100"]["lengths"][0])

        if "ud" in raw_data[lang] and system in raw_data[lang]["ud"] and [lang, "ud"] not in main_table_exclude:
            preds_main_results_indicies.append(raw_data[lang]["ud"][system][0])
            trues_main_results_indicies.append(raw_data[lang]["ud"]["true_indices"])
            lengths_main_results.append(raw_data[lang]["ud"]["lengths"][0])

        if (
            "ersatz" in raw_data[lang]
            and system in raw_data[lang]["ersatz"]
            and [lang, "ersatz"] not in main_table_exclude
        ):
            preds_main_results_indicies.append(raw_data[lang]["ersatz"][system][0])
            trues_main_results_indicies.append(raw_data[lang]["ersatz"]["true_indices"])
            lengths_main_results.append(raw_data[lang]["ersatz"]["lengths"][0])

        raw_data[lang]["main_table_mean"][system] = preds_main_results_indicies

        if "true_indices" in raw_data[lang]["main_table_mean"]:
            assert (
                raw_data[lang]["main_table_mean"]["true_indices"] == trues_main_results_indicies
            ), f"{lang} {system}, {[len(i) for i in trues_main_results_indicies]}, {[len(i) for i in raw_data[lang]['main_table_mean']['true_indices']]}"
        else:
            raw_data[lang]["main_table_mean"]["true_indices"] = trues_main_results_indicies

        if "lengths" in raw_data[lang]["main_table_mean"]:
            assert (
                raw_data[lang]["main_table_mean"]["lengths"] == lengths_main_results
            ), f"{lang} {system} {raw_data[lang]['main_table_mean']['lengths']} {lengths_main_results}"
        else:
            raw_data[lang]["main_table_mean"]["lengths"] = lengths_main_results


raw_data = {k: dict(v) for k, v in raw_data.items()}

with open(DATA_DIR / "main_all_raw_data.pkl", "wb") as f:
    pickle.dump(raw_data, f)

val_results = {k: dict(v) for k, v in val_results.items()}

with open(DATA_DIR / "main_all_val_results.pkl", "wb") as f:
    pickle.dump(val_results, f)
