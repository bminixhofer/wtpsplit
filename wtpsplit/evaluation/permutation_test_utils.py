import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def compute_prf(true_values, predicted_values, num_docs):

    f1 = 0
    r = 0
    p = 0

    for true, pred in zip(true_values, predicted_values):

        TP = np.sum((pred == 1) & (true == 1))
        FP = np.sum((pred == 1) & (true == 0))
        FN = np.sum((pred == 0) & (true == 1))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        p += precision
        r += recall
        f1 += 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    p /= num_docs
    r /= num_docs
    f1 /= num_docs

    return p, r, f1


def test_func(x, y, true, num_docs):
    p_x, r_x, f1_x = compute_prf(true, x, num_docs)
    p_y, r_y, f1_y = compute_prf(true, y, num_docs)

    diff_p = np.abs(p_x - p_y)
    diff_r = np.abs(r_x - r_y)
    diff_f1 = np.abs(f1_x - f1_y)

    return diff_p, diff_r, diff_f1


def permutation_test_single_round(x, y, true, y_lengths, num_docs, flips):
    sample_x = [np.where(flips[:m], y[i], x[i]) for i, m in enumerate(y_lengths)]
    sample_y = [np.where(flips[:m], x[i], y[i]) for i, m in enumerate(y_lengths)]

    return test_func(sample_x, sample_y, true, num_docs)


def permutation_test(
    x,
    y,
    true,
    num_rounds=10000,
):

    # print(num_rounds)

    x_lengths = [len(i) for i in x]
    y_lengths = [len(i) for i in y]

    for i, j in zip(x_lengths, y_lengths):
        assert i == j

    p_at_least_as_extreme = 0.0
    r_at_least_as_extreme = 0.0
    f_at_least_as_extreme = 0.0

    num_docs = len(true)

    p_reference_stat, r_reference_stat, f_reference_stat = test_func(x, y, true, num_docs)

    flips = np.random.randint(2, size=(num_rounds, max(y_lengths)))

    with Pool(5) as pool:
        results = list(
            pool.starmap(
                permutation_test_single_round,
                tqdm(
                    [(x, y, true, y_lengths, num_docs, flips[i]) for i in range(num_rounds)],
                    total=num_rounds,
                ),
            ),
        )

    for diff_p, diff_r, diff_f in results:
        if diff_p > p_reference_stat or np.isclose(diff_p, p_reference_stat):
            p_at_least_as_extreme += 1.0

        if diff_r > r_reference_stat or np.isclose(diff_r, r_reference_stat):
            r_at_least_as_extreme += 1.0

        if diff_f > f_reference_stat or np.isclose(diff_f, f_reference_stat):
            f_at_least_as_extreme += 1.0

    return (
        results,
        p_at_least_as_extreme / num_rounds,
        r_at_least_as_extreme / num_rounds,
        f_at_least_as_extreme / num_rounds,
    )


def print_latex(df, systems, all_systems_mapping, results, filename):

    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w") as f:

        latex = df.to_latex(float_format="%.3f", escape=False, columns=systems)
        while "  " in latex:
            latex = latex.replace("  ", " ")
        latex = latex.replace("-100.000", "-")

        for system, system_name in all_systems_mapping.items():
            latex = latex.replace(system, system_name)

        for system, system_name in all_systems_mapping.items():
            if system in results:
                latex += "\n"
                latex += f"% {system_name}: {round(results[system], 3)}"

        f.write(latex)


def reverse_where(true_indices, pred_indices, lengths):

    y_true_all = []
    y_pred_all = []

    for true, pred, length in zip(true_indices, pred_indices, lengths):

        y_true = np.zeros(length)
        y_true[true] = 1
        y_pred = np.zeros(length)
        y_pred[pred] = 1

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    return y_true_all, y_pred_all
