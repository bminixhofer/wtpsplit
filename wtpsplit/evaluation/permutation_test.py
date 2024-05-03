import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from wtpsplit.evaluation.permutation_test_utils import compute_prf, permutation_test, print_latex, reverse_where
from tqdm import tqdm

ALL_DIR = Path("data/permutation-test-data/03-05")

raw_data = defaultdict(lambda: defaultdict(dict))
val_results = defaultdict(lambda: defaultdict(dict))


DATA_DIR = ALL_DIR / "data"
LATEX_DIR = ALL_DIR / "results"

for file in DATA_DIR.glob("*IDX.json"):
    if "12l" in file.stem:
        continue
    model = str(file.stem)[:-4]
    with open(file, "r") as f:
        data = json.load(f)
    for lang in data.keys():
        for dataset in data[lang].keys():
            for model_type in data[lang][dataset].keys():
                if model_type == "true_indices" or model_type == "length":
                    continue
                raw_data[lang][dataset][model + "-" + model_type] = data[lang][dataset][model_type]

            if "true_indicies" in raw_data[lang][dataset]:
                assert raw_data[lang][dataset]["true_indices"] == data[lang][dataset]["true_indices"]
            else:
                raw_data[lang][dataset]["true_indices"] = data[lang][dataset]["true_indices"]

            if "length" in raw_data[lang][dataset]:
                assert raw_data[lang][dataset]["length"] == data[lang][dataset]["length"]
            else:
                raw_data[lang][dataset]["length"] = data[lang][dataset]["length"]


for file in DATA_DIR.glob("*.json"):
    if file.stem.endswith("IDX"):
        continue

    with open(file, "r") as f:
        data = json.load(f)

    model = file.stem

    for lang in data.keys():
        for dataset in data[lang].keys():
            for model_type in data[lang][dataset].keys():
                val_results[lang][dataset][model + "-" + model_type] = data[lang][dataset][model_type]


for lang in tqdm(raw_data.keys()):
    for dataset in raw_data[lang].keys():

        systems = sorted(list(raw_data[lang][dataset].keys()))
        systems.remove("true_indices")
        systems.remove("length")

        systems = [s for s in systems if val_results[lang][dataset][s] is not None]

        num_systems = len(systems)

        p_pvalues = pd.DataFrame(-100 + np.zeros((num_systems, num_systems)), index=systems, columns=systems)
        r_pvalues = pd.DataFrame(-100 + np.zeros((num_systems, num_systems)), index=systems, columns=systems)
        f_pvalues = pd.DataFrame(-100 + np.zeros((num_systems, num_systems)), index=systems, columns=systems)

        total_permutation_tests = num_systems * (num_systems - 1) // 2

        for model in systems:
            true_indices = raw_data[lang][dataset]["true_indices"]
            pred_indices = raw_data[lang][dataset][model]
            if pred_indices is None:
                continue
            length = raw_data[lang][dataset]["length"]
            y_true, y_pred = reverse_where(true_indices, pred_indices, length)
            _, _, f1 = compute_prf(y_true, y_pred)
            print(f"{lang} {dataset} {model} F1: {f1}")
            assert np.allclose(
                f1, val_results[lang][dataset][model]
            ), f" MISMATCH! {lang} {dataset} {model} F1: {f1} intrinsic_py_out: {val_results[lang][dataset][model]}"

        for i in range(num_systems):
            for j in range(i + 1, num_systems):
                true_indices = raw_data[lang][dataset]["true_indices"]
                pred1_indices = raw_data[lang][dataset][systems[i]]
                pred2_indices = raw_data[lang][dataset][systems[j]]
                length = raw_data[lang][dataset]["length"]
                y_true, y_pred1 = reverse_where(true_indices, pred1_indices, length)
                y_true, y_pred2 = reverse_where(true_indices, pred2_indices, length)

                p_pvalue, r_pvalue, f_pvalue = permutation_test(
                    y_pred1,
                    y_pred2,
                    y_true,
                    num_rounds=10000,
                )

                p_pvalues.at[systems[i], systems[j]] = p_pvalue
                r_pvalues.at[systems[i], systems[j]] = r_pvalue
                f_pvalues.at[systems[i], systems[j]] = f_pvalue

        print_latex(p_pvalues, systems, LATEX_DIR / f"{lang}_{dataset}_p_pvalues.tex")
        print_latex(r_pvalues, systems, LATEX_DIR / f"{lang}_{dataset}_r_pvalues.tex")
        print_latex(f_pvalues, systems, LATEX_DIR / f"{lang}_{dataset}_f_pvalues.tex")

print("All tests passed!")
