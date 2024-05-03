import numpy as np
from tqdm import tqdm


def compute_prf(true_values, predicted_values):

    TP = np.sum((predicted_values == 1) & (true_values == 1))
    FP = np.sum((predicted_values == 1) & (true_values == 0))
    FN = np.sum((predicted_values == 0) & (true_values == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def test_func(x, y, true):
    p_x, r_x, f1_x = compute_prf(true, x)
    p_y, r_y, f1_y = compute_prf(true, y)

    diff_p = np.abs(p_x - p_y)
    diff_r = np.abs(r_x - r_y)
    diff_f1 = np.abs(f1_x - f1_y)

    return diff_p, diff_r, diff_f1


def permutation_test(
    x,
    y,
    true,
    num_rounds=10000,
    seed=None,
):

    rng = np.random.RandomState(seed)

    m, n = len(x), len(y)

    if m != n:
        raise ValueError(
            f"x and y must have the same" f" length if `paired=True`, but they had lengths {m} and {n} respectively."
        )

    sample_x = np.empty(m)
    sample_y = np.empty(n)

    p_at_least_as_extreme = 0.0
    r_at_least_as_extreme = 0.0
    f_at_least_as_extreme = 0.0

    p_reference_stat, r_reference_stat, f_reference_stat = test_func(x, y, true)

    # this loop currently takes 30 seconds. Probably all because of test_func although that is already quite optimized
    # maybe we can do these rounds in parallel

    for i in range(num_rounds):
        flip = rng.randn(m) > 0.0

        # for i, f in enumerate(flip):
        #     if f:
        #         sample_x[i], sample_y[i] = y[i], x[i]
        #     else:
        #         sample_x[i], sample_y[i] = x[i], y[i]

        sample_x = np.where(flip, y, x)
        sample_y = np.where(flip, x, y)

        diff_p, diff_r, diff_f = test_func(sample_x, sample_y, true)

        if diff_p > p_reference_stat or np.isclose(diff_p, p_reference_stat):
            p_at_least_as_extreme += 1.0

        if diff_r > r_reference_stat or np.isclose(diff_r, r_reference_stat):
            r_at_least_as_extreme += 1.0

        if diff_f > f_reference_stat or np.isclose(diff_f, f_reference_stat):
            f_at_least_as_extreme += 1.0

    return (
        p_at_least_as_extreme / num_rounds,
        r_at_least_as_extreme / num_rounds,
        f_at_least_as_extreme / num_rounds,
    )


def print_latex(df, systems, filename):

    with open(filename, "w") as f:

        latex = df.to_latex(float_format="%.3f", escape=False, columns=systems)
        while "  " in latex:
            latex = latex.replace("  ", " ")
        latex = latex.replace("-100.000", "-")
        f.write(latex)


def reverse_where(true_indices, pred_indices, length):
    y_true = np.zeros(length)
    y_true[true_indices] = 1
    y_pred = np.zeros(length)
    try:
        y_pred[pred_indices] = 1
    except:
        print(f"pred_indices: {pred_indices}")
        raise Exception
    return y_true, y_pred
