import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from wtpsplit.evaluation.permutation_test_utils import compute_prf, permutation_test, print_latex, reverse_where

parser = argparse.ArgumentParser()

parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--table", type=str, required=True)

args = parser.parse_args()

ALL_DIR = Path("../data/permutation-test-data/")

raw_data = defaultdict(lambda: defaultdict(dict))
val_results = defaultdict(lambda: defaultdict(dict))


DATA_DIR = ALL_DIR / f"all-stat-test-data/{args.table}"
LATEX_DIR = ALL_DIR / "p-values"
RESULTS_DATA_DIR = ALL_DIR / "results_data"

spacy_langs = open("../data/permutation-test-data/all-stat-test-data/spacy_m_langs.txt").read().splitlines()

with open(DATA_DIR / f"{args.table}_raw_data.pkl", "rb") as f:
    raw_data = pickle.load(f)

with open(DATA_DIR / f"{args.table}_val_results.pkl", "rb") as f:
    val_results = pickle.load(f)

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

lora_filter_data = [
    ["ceb", "ted2020-corrupted-asr"],
    ["et", "short-sequences"],
    ["et", "short-sequences-corrupted-asr"],
    ["ga", "ted2020-corrupted-asr"],
    ["ha", "ted2020-corrupted-asr"],
    ["ig", "ted2020-corrupted-asr"],
    ["kk", "ud"],
    ["ky", "ted2020-corrupted-asr"],
    ["la", "ted2020-corrupted-asr"],
    ["mg", "ted2020-corrupted-asr"],
    ["mr", "ud"],
    ["mt", "ted2020-corrupted-asr"],
    ["pa", "ted2020-corrupted-asr"],
    ["ta", "ud"],
    ["tg", "ted2020-corrupted-asr"],
    ["en-de", "short-sequences"],
    ["en-de", "short-sequences-corrupted-asr"],
    ["sr", "short-sequences"],
    ["sr", "short-sequences-corrupted-asr"],
    ["sl", "short-sequences"],
    ["sl", "short-sequences-corrupted-asr"],
]

for dataset in raw_data[args.lang].keys():

    systems = list(all_systems_mapping.keys()).copy()

    if [args.lang, dataset] in lora_filter_data:
        systems.remove("xlmr-3l-v4_LL_lora-v2_ep30_s10k_b512_s64_u0.5_k10-t")
    else:
        systems.remove("xlmr-3l-v4_LL_lora-v2_ep30_s10k_b512_s64_u0.5_k10-u")

    systems = [
        s for s in systems if s in val_results[args.lang][dataset] and val_results[args.lang][dataset][s] is not None
    ]

    systems = sorted(systems, key=lambda x: val_results[args.lang][dataset][x], reverse=True)

    num_systems = len(systems)

    p_pvalues = pd.DataFrame(-100 + np.zeros((num_systems, num_systems)), index=systems, columns=systems)
    r_pvalues = pd.DataFrame(-100 + np.zeros((num_systems, num_systems)), index=systems, columns=systems)
    f_pvalues = pd.DataFrame(-100 + np.zeros((num_systems, num_systems)), index=systems, columns=systems)

    all_diffs = {system1: {} for system1 in systems}

    total_permutation_tests = num_systems * (num_systems - 1) // 2

    for model in systems:

        true_indices = raw_data[args.lang][dataset]["true_indices"]
        pred_indices = raw_data[args.lang][dataset][model]
        if pred_indices is None:
            continue
        lengths = raw_data[args.lang][dataset]["lengths"]
        y_true, y_pred = reverse_where(true_indices, pred_indices, lengths)
        num_docs = len(y_true)

        _, _, f1 = compute_prf(y_true, y_pred, num_docs)

        assert np.allclose(
            f1, val_results[args.lang][dataset][model]
        ), f" MISMATCH! {args.lang} {dataset} {model} F1: {f1} intrinsic_py_out: {val_results[args.lang][dataset][model]}"

    for i in range(num_systems):
        for j in range(i + 1, num_systems):
            true_indices = raw_data[args.lang][dataset]["true_indices"]
            pred1_indices = raw_data[args.lang][dataset][systems[i]]
            pred2_indices = raw_data[args.lang][dataset][systems[j]]
            lengths = raw_data[args.lang][dataset]["lengths"]
            y_true, y_pred1 = reverse_where(true_indices, pred1_indices, lengths)
            y_true, y_pred2 = reverse_where(true_indices, pred2_indices, lengths)

            diffs, p_pvalue, r_pvalue, f_pvalue = permutation_test(
                y_pred1,
                y_pred2,
                y_true,
                num_rounds=10000,
            )

            p_pvalues.at[systems[i], systems[j]] = p_pvalue
            r_pvalues.at[systems[i], systems[j]] = r_pvalue
            f_pvalues.at[systems[i], systems[j]] = f_pvalue

            all_diffs[systems[i]][systems[j]] = diffs

    print_latex(
        f_pvalues,
        systems,
        all_systems_mapping,
        val_results[args.lang][dataset],
        LATEX_DIR / f"{dataset}/{args.lang}_f.tex",
    )

    saving_data = {
        "p_pvalues": p_pvalues,
        "r_pvalues": r_pvalues,
        "f_pvalues": f_pvalues,
        "all_diffs": all_diffs,
    }

    if not (RESULTS_DATA_DIR / dataset).exists():
        (RESULTS_DATA_DIR / dataset).mkdir()

    with open(RESULTS_DATA_DIR / f"{dataset}/{args.lang}_data.pkl", "wb") as f:
        pickle.dump(saving_data, f)

    print(f"Finished {args.lang} {dataset}")

print("All validation tests passed and significance tests done!")
