import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import os

"""
Plotting utilities for representation-benchmark analysis.

This module provides scatter plots that relate representation similarity
to (a) model performance change and (b) structural tree difference.
"""

def plot_similarity_performance(
    df,
    representation_name,
    save_results=False,
    savepath=None,
    *,
    fold_col: str = "fold_idx",
    color_by_fold: bool = False,
    use_rep_distance: bool = True,
    use_accuracy_drop: bool = True,
):
    """
    Plot representation similarity vs. predictive performance.

    Parameters
    ----------
    df : pandas.DataFrame
        Benchmark table containing `sim_<representation_name>` and performance columns.
    representation_name : str
        Representation identifier used to build similarity column name.
    save_results : bool, default=False
        If True, save the figure to `savepath`; otherwise display it.
    savepath : str | None
        Output image path used when `save_results=True`.
    use_rep_distance : bool, default=True
        If True, transform similarity to distance via `1 - sim`.
    use_accuracy_drop : bool, default=True
        If True, y-axis is `performance_base - performance_perturbed`;
        otherwise y-axis is `performance_perturbed`.

    Returns
    -------
    tuple
        (figure, pearson_r, pearson_pvalue)
    """
    fig = plt.figure(figsize=(6, 4))

    sim_col = f"sim_{representation_name}"
    x_raw = df[sim_col]
    y_raw = _get_accuracy_drop(df) if use_accuracy_drop else df["performance_perturbed"]

    x_all = _to_rep_distance(x_raw) if use_rep_distance else np.asarray(x_raw, dtype=float)
    x_all, y_all = _clean_xy(x_all, y_raw)

    if "perturbation" in df.columns and "intensity" in df.columns:
        perturbations = sorted(df["perturbation"].dropna().unique().tolist(), key=lambda v: str(v))
        intensities = df["intensity"].dropna().astype(float)
        min_i = float(intensities.min()) if len(intensities) else 0.0
        max_i = float(intensities.max()) if len(intensities) else 1.0

        cmap = plt.get_cmap("tab10")
        color_map = {p: cmap(i % 10) for i, p in enumerate(perturbations)}

        for p in perturbations:
            dpp = df[df["perturbation"] == p]
            uniq_ints = sorted([float(v) for v in dpp["intensity"].dropna().unique().tolist()])
            labeled = False
            for inten in uniq_ints:
                dppi = dpp[dpp["intensity"].astype(float) == float(inten)]
                xx_raw = dppi[sim_col]
                yy_raw = _get_accuracy_drop(dppi) if use_accuracy_drop else dppi["performance_perturbed"]
                xx = _to_rep_distance(xx_raw) if use_rep_distance else np.asarray(xx_raw, dtype=float)
                xx, yy = _clean_xy(xx, yy_raw)
                if len(xx) == 0:
                    continue
                alpha = _alpha_from_intensity(float(inten), min_i, max_i, a_min=0.15, a_max=1.0)
                plt.scatter(xx, yy, s=18, color=color_map[p], alpha=alpha, label=str(p) if not labeled else None)
                labeled = True
    else:
        plt.scatter(x_all, y_all, color="lightskyblue", alpha=0.7, s=18)

    if len(x_all) >= 2 and np.std(x_all) > 0 and np.std(y_all) > 0:
        r, pval = pearsonr(x_all, y_all)
        m, b = np.polyfit(x_all, y_all, 1)
        xs = np.linspace(np.min(x_all), np.max(x_all), 200)
        plt.plot(xs, m * xs + b, color="red", linewidth=2)
    else:
        r, pval = np.nan, np.nan

    plt.xlabel("Δ Representation Similarity" if use_rep_distance else "Representation Similarity to Base")
    plt.ylabel("Δ Accuracy" if use_accuracy_drop else "Perturbed Accuracy")
    plt.title(f"Similarity vs Performance — {representation_name}\n"
              f"n={len(x_all)}, r={r:.2f}, p={pval:.3f}")
    plt.tight_layout()

    if "perturbation" in df.columns:
        leg = plt.legend(fontsize=8, frameon=False)
        for h in leg.legendHandles:
            h.set_alpha(1.0)

    if save_results and savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig, float(r) if np.isfinite(r) else float("nan"), float(pval) if np.isfinite(pval) else float("nan")

def plot_similarity_structural_difference(
    df,
    representation_name,
    save_results=False,
    savepath=None,
    *,
    fold_col: str = "fold_idx",
    color_by_fold: bool = False,
    use_rep_distance: bool = True,
):
    """
    Plot representation similarity vs. structural difference (tree edit distance).

    Parameters
    ----------
    df : pandas.DataFrame
        Benchmark table containing `sim_<representation_name>` and structural difference columns.
    representation_name : str
        Representation identifier used to build similarity column name.
    save_results : bool, default=False
        If True, save the figure to `savepath`; otherwise display it.
    savepath : str | None
        Output image path used when `save_results=True`.
    use_rep_distance : bool, default=True
        If True, transform similarity to distance via `1 - sim`.

    Returns
    -------
    tuple
        (figure, pearson_r, pearson_pvalue)
    """
    fig = plt.figure(figsize=(6, 4))

    sim_col = f"sim_{representation_name}"
    x_raw = df[sim_col]
    x_all = _to_rep_distance(x_raw) if use_rep_distance else np.asarray(x_raw, dtype=float)
    x_all, y_all = _clean_xy(x_all, df["structural_difference"])

    if "perturbation" in df.columns and "intensity" in df.columns:
        perturbations = sorted(df["perturbation"].dropna().unique().tolist(), key=lambda v: str(v))
        intensities = df["intensity"].dropna().astype(float)
        min_i = float(intensities.min()) if len(intensities) else 0.0
        max_i = float(intensities.max()) if len(intensities) else 1.0

        cmap = plt.get_cmap("tab10")
        color_map = {p: cmap(i % 10) for i, p in enumerate(perturbations)}

        for p in perturbations:
            dpp = df[df["perturbation"] == p]
            uniq_ints = sorted([float(v) for v in dpp["intensity"].dropna().unique().tolist()])
            labeled = False
            for inten in uniq_ints:
                dppi = dpp[dpp["intensity"].astype(float) == float(inten)]
                xx_raw = dppi[sim_col]
                xx = _to_rep_distance(xx_raw) if use_rep_distance else np.asarray(xx_raw, dtype=float)
                xx, yy = _clean_xy(xx, dppi["structural_difference"])
                if len(xx) == 0:
                    continue
                alpha = _alpha_from_intensity(float(inten), min_i, max_i, a_min=0.15, a_max=1.0)
                plt.scatter(xx, yy, s=18, color=color_map[p], alpha=alpha, label=str(p) if not labeled else None)
                labeled = True
    else:
        plt.scatter(x_all, y_all, color="mediumspringgreen", alpha=0.7, s=18)

    if len(x_all) >= 2 and np.std(x_all) > 0 and np.std(y_all) > 0:
        r, pval = pearsonr(x_all, y_all)
        m, b = np.polyfit(x_all, y_all, 1)
        xs = np.linspace(np.min(x_all), np.max(x_all), 200)
        plt.plot(xs, m * xs + b, color="red", linewidth=2)
    else:
        r, pval = np.nan, np.nan

    plt.xlabel("Δ Representation Similarity" if use_rep_distance else "Representation Similarity to Base")
    plt.ylabel("Δ Structure")
    plt.title(f"Similarity vs Structural Difference — {representation_name}\n"
              f"n={len(x_all)}, r={r:.2f}, p={pval:.3f}")
    plt.tight_layout()

    if "perturbation" in df.columns:
        leg = plt.legend(fontsize=8, frameon=False)
        for h in leg.legendHandles:
            h.set_alpha(1.0)

    if save_results and savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig, float(r) if np.isfinite(r) else float("nan"), float(pval) if np.isfinite(pval) else float("nan")


def _clean_xy(x, y):
    """Return finite-only x/y pairs as float numpy arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def _alpha_from_intensity(intensity: float, min_i: float, max_i: float, a_min: float = 0.15, a_max: float = 1.0) -> float:
    """Map perturbation intensity linearly to alpha in [a_min, a_max]."""
    if not np.isfinite(intensity):
        return a_min
    if max_i <= min_i:
        return a_max
    t = (float(intensity) - float(min_i)) / (float(max_i) - float(min_i))
    t = float(np.clip(t, 0.0, 1.0))
    return a_min + t * (a_max - a_min)

def _to_rep_distance(sim):
    """Convert similarity to distance using `1 - similarity`."""
    sim = np.asarray(sim, dtype=float)
    return 1.0 - sim

def _get_accuracy_drop(df):
    """
    Compute accuracy drop from supported base-performance columns.

    Expected base column precedence:
    `performance_base`, `performance_base_tree`, `acc_base`.
    """
    for c in ("performance_base", "performance_base_tree", "acc_base"):
        if c in df.columns:
            return np.asarray(df[c], dtype=float) - np.asarray(df["performance_perturbed"], dtype=float)
    raise KeyError("Base-Performance column missing. Expected one of: performance_base, performance_base_tree, acc_base")
