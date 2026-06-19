import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.manifold import MDS
import os
import pandas as pd
import seaborn as sns
from itertools import combinations


metrics = ["mcc"]
metrics_written = ["MCC"]

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
    fig = plt.figure(figsize=(7, 5))

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
    plt.ylabel("Δ MCC" if use_accuracy_drop else "Perturbed MCC")
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
    else:
        pass#plt.show()

    return fig, float(r) if np.isfinite(r) else float("nan"), float(pval) if np.isfinite(pval) else float("nan")

def plot_similarity_fi_difference(
    df,
    representation_name,
    save_results=False,
    savepath=None,
    *,
    fold_col: str = "fold_idx",
    color_by_fold: bool = False,
    use_rep_distance: bool = True,
):
    fig = plt.figure(figsize=(7, 5))

    sim_col = f"sim_{representation_name}"
    x_raw = df[sim_col]
    x_all = _to_rep_distance(x_raw) if use_rep_distance else np.asarray(x_raw, dtype=float)
    x_all, y_all = _clean_xy(x_all, df["feature_importance_difference"])

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
                xx, yy = _clean_xy(xx, dppi["feature_importance_difference"])
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
    plt.ylabel("Δ Feature Importance")
    plt.title(f"Similarity vs Feature Importance Difference — {representation_name}\n"
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
    else:
        pass#plt.show()

    return fig, float(r) if np.isfinite(r) else float("nan"), float(pval) if np.isfinite(pval) else float("nan")

def plot_and_save_mds(distance_matrix, accuracies, name, savepath, seed):
    if np.all(np.isnan(accuracies)):
        color_data = np.zeros(len(accuracies))
    else:
        min_acc = np.nanmin(accuracies)
        color_data = np.nan_to_num(accuracies, nan=min_acc)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed, normalized_stress='auto', n_init=4)
    coords = mds.fit_transform(distance_matrix)

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=color_data, cmap='cividis', 
                        edgecolors='black', linewidth=0.5, alpha=0.8, s=60)
    
    plt.colorbar(sc, label='Individual Tree OOB MCC')
    plt.title(f"MDS Tree Projection: {name}")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plots_dir = os.path.join(savepath, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    filename = os.path.join(
        plots_dir,
        f"mds_plot_{name.replace(' ', '_').lower()}_{seed}.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_vs_subforest_size(path, dataset_name, dataset_name_written, output_dir=None):
    df = _read_subforest_file(path)
    results = _calculate_subforest_metrics(df)

    
    for metric, metric_written in zip(metrics, metrics_written):
        ff = results["full_forest"][metric].iloc[0]
        random_df = results["random"][["subforest_size", metric]].copy()
        random_df["representation"] = "Random"
        top_oob_acc_df = results["top_oob_acc"][["subforest_size", metric]].copy()
        top_oob_acc_df["representation"] = "Top OOB ACC"
        top_oob_mcc_df = results["top_oob_mcc"][["subforest_size", metric]].copy()
        top_oob_mcc_df["representation"] = "Top OOB MCC"

        clusterings = results["representations"]["selection_strategy"].dropna().unique()

        marker_mapping = {
            "Random": "o",
            "Top OOB ACC": "s",
            "Top OOB MCC": "P",
            "Feature Graph": "D",
            "Tree Descriptor": "^",
            "Leaf Profile": "v",
            "Topological Forest": "<",
            "INDTree": ">"
        }

        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        palette = sns.color_palette("colorblind", len(marker_mapping))

        for clustering in clusterings:
            plt.figure(figsize=(7, 5), dpi=300)
            rep_df = results["representations"]
            rep_df = rep_df[rep_df["selection_strategy"] == clustering][["representation", "subforest_size", metric]].copy()
            plot_df = pd.concat([rep_df, random_df, top_oob_acc_df, top_oob_mcc_df], ignore_index=True)
            
            sns.lineplot(
                data=plot_df,
                x="subforest_size",
                y=metric,
                hue="representation",
                style="representation",
                palette=palette,
                dashes=False,
                markers=marker_mapping,
                linewidth=2,
                markersize=5
            )

            plt.axhline(
                y=ff,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Full Forest"
            )

            plt.title(f"Dataset: {dataset_name_written}\nSelection strategy: {clustering}")
            plt.xlabel("Subforest size")
            plt.ylabel(metric_written)
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

            plt.figlegend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=4,
                frameon=False,
                fontsize=10,
                columnspacing=1.0,
                handletextpad=0.4
            )
            plt.subplots_adjust(bottom=0.24)
            sns.despine()
            #plt.show()
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_{metric}_{clustering}_vs_subforest_size.png"), bbox_inches="tight")
            plt.close()

def plot_feature_importance_stability_vs_subforest_size(
    path,
    dataset_name,
    dataset_name_written,
    method="spearman",
    output_dir=None
):
    df = _read_subforest_file(path)
    stability_df = _calculate_feature_importance_stability(df, method=method)

    ff = stability_df[stability_df["representation"] == "Full Forest"]["stability"].iloc[0]

    # baselines
    baseline_df = stability_df[stability_df["selection_strategy"].isna()].copy()
    random_df = baseline_df[baseline_df["representation"] == "Random"].copy()
    random_df["representation"] = "Random"
    top_oob_acc_df = baseline_df[baseline_df["representation"] == "Top OOB ACC"].copy()
    top_oob_acc_df["representation"] = "Top OOB ACC"
    top_oob_mcc_df = baseline_df[baseline_df["representation"] == "Top OOB MCC"].copy()
    top_oob_mcc_df["representation"] = "Top OOB MCC"

    # Clusterings from representations
    clusterings = stability_df["selection_strategy"].dropna().unique()
    marker_mapping = {
        "Random": "o",
        "Top OOB ACC": "s",
        "Top OOB MCC": "P",
        "Feature Graph": "D",
        "Tree Descriptor": "^",
        "Leaf Profile": "v",
        "Topological Forest": "<",
        "INDTree": ">"
    }

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("colorblind", len(marker_mapping))

    os.makedirs(output_dir, exist_ok=True)

    method_label = "Spearman" if method.lower() == "spearman" else "Kendall's W"

    for clustering in clusterings:
        plt.figure(figsize=(7, 5), dpi=300)

        rep_df = stability_df[stability_df["selection_strategy"] == clustering][
            ["representation", "subforest_size", "stability"]
        ].copy()

        plot_df = pd.concat([rep_df, random_df, top_oob_acc_df, top_oob_mcc_df], ignore_index=True)

        sns.lineplot(
            data=plot_df,
            x="subforest_size",
            y="stability",
            hue="representation",
            style="representation",
            palette=palette,
            dashes=False,
            markers=marker_mapping,
            linewidth=2,
            markersize=5
        )

        plt.axhline(
                y=ff,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Full Forest"
            )

        plt.title(f"Dataset: {dataset_name_written}\nSelection strategy: {clustering}")
        plt.xlabel("Subforest size")
        plt.ylabel(f"Feature-Importance Stability ({method_label})")

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if ax.legend_ is not None:
            ax.legend_.remove()

        plt.figlegend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            frameon=False,
            fontsize=10,
            columnspacing=1.0,
            handletextpad=0.4
        )
        plt.subplots_adjust(bottom=0.24)
        sns.despine()

        out_file = os.path.join(
            output_dir,
            f"{dataset_name}_feature_importance_{method}_{clustering}_vs_subforest_size.png"
        )
        plt.savefig(out_file, bbox_inches="tight")
        plt.close()

def print_clustering_differences_vs_baseline(
    dataset_paths,
    baseline_clustering,
    output_csv=None,
    average_across_subforest_sizes=False,
    output_csv_avg=None
):
    metric_names = {
        "acc": "Accuracy",
        "macro_f1": "Macro F1",
        "mcc": "MCC",
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC",
        "agreement_with_full_forest": "Agreement with Full Forest"
    }

    def _as_txt_path(path):
        if path is None:
            return None
        base, _ = os.path.splitext(path)
        return base + ".txt"

    main_lines = []
    avg_lines = []

    def _emit(line="", bucket=None):
        if bucket is not None:
            bucket.append(line)

    rep_parts = []

    # 1) Collect representation results from all datasets
    for dataset_id, path in enumerate(dataset_paths):
        df = _read_subforest_file(path)
        results = _calculate_subforest_metrics(df)

        rep_df = results["representations"]
        if rep_df.empty:
            continue

        tmp = rep_df[["representation", "selection_strategy", "subforest_size"] + metrics].copy()
        tmp["dataset_id"] = dataset_id
        rep_parts.append(tmp)

    if not rep_parts:
        raise ValueError("No representation data found in the provided dataset_paths.")

    rep_all = pd.concat(rep_parts, ignore_index=True)

    # 2) Average within each dataset across representations
    per_dataset = (
        rep_all
        .groupby(["dataset_id", "selection_strategy", "subforest_size"])[metrics]
        .mean()
        .reset_index()
    )

    # 3) Average across datasets
    avg_by_clustering = (
        per_dataset
        .groupby(["selection_strategy", "subforest_size"])[metrics]
        .mean()
        .reset_index()
    )

    available_clusterings = sorted(avg_by_clustering["selection_strategy"].dropna().unique().tolist())
    if baseline_clustering not in available_clusterings:
        raise ValueError(
            f"Baseline clustering '{baseline_clustering}' not found. "
            f"Available: {available_clusterings}"
        )

    baseline_df = (
        avg_by_clustering[avg_by_clustering["selection_strategy"] == baseline_clustering]
        .set_index("subforest_size")
        .sort_index()
    )

    rows = []
    for clustering in available_clusterings:
        if clustering == baseline_clustering:
            continue

        other_df = (
            avg_by_clustering[avg_by_clustering["selection_strategy"] == clustering]
            .set_index("subforest_size")
            .sort_index()
        )

        common_sizes = baseline_df.index.intersection(other_df.index)
        for size in common_sizes:
            for metric in metrics:
                delta = other_df.loc[size, metric] - baseline_df.loc[size, metric]
                rows.append({
                    "baseline": baseline_clustering,
                    "selection_strategy": clustering,
                    "subforest_size": int(size),
                    "metric": metric,
                    "delta_vs_baseline": float(delta)
                })

    delta_df = pd.DataFrame(rows).sort_values(["subforest_size", "metric", "selection_strategy"])

    # 4) Print per-subforest-size report
    _emit(f"\n=== Differences vs baseline selection strategy: {baseline_clustering} ===", main_lines)
    for size in sorted(delta_df["subforest_size"].unique()):
        _emit(f"\nSubforest size = {size}", main_lines)
        for metric in metrics:
            block = delta_df[
                (delta_df["subforest_size"] == size) &
                (delta_df["metric"] == metric)
            ]
            if block.empty:
                continue

            _emit(f"  {metric_names[metric]}:", main_lines)
            for _, r in block.iterrows():
                d = r["delta_vs_baseline"]
                status = "better" if d > 0 else ("worse" if d < 0 else "equal")
                _emit(f"    {r['selection_strategy']}: {d:+.4f} ({status})", main_lines)

    if output_csv is not None:
        out_txt = _as_txt_path(output_csv)
        os.makedirs(os.path.dirname(out_txt), exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(main_lines) + "\n")

    # 5) Optional: additionally average across subforest sizes
    if average_across_subforest_sizes:
        delta_avg_df = (
            delta_df
            .groupby(["baseline", "selection_strategy", "metric"], as_index=False)["delta_vs_baseline"]
            .mean()
            .sort_values(["metric", "selection_strategy"])
        )

        _emit(f"\n=== Averaged across subforest sizes (vs {baseline_clustering}) ===", avg_lines)
        for metric in metrics:
            block = delta_avg_df[delta_avg_df["metric"] == metric]
            if block.empty:
                continue
            _emit(f"  {metric_names[metric]}:", avg_lines)
            for _, r in block.iterrows():
                d = r["delta_vs_baseline"]
                status = "better" if d > 0 else ("worse" if d < 0 else "equal")
                _emit(f"    {r['selection_strategy']}: {d:+.4f} ({status})", avg_lines)

        if output_csv_avg is not None:
            out_txt_avg = _as_txt_path(output_csv_avg)
            os.makedirs(os.path.dirname(out_txt_avg), exist_ok=True)
            with open(out_txt_avg, "w", encoding="utf-8") as f:
                f.write("\n".join(avg_lines) + "\n")

    return delta_df


def _clean_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def _alpha_from_intensity(intensity: float, min_i: float, max_i: float, a_min: float = 0.15, a_max: float = 1.0) -> float:
    if not np.isfinite(intensity):
        return a_min
    if max_i <= min_i:
        return a_max
    t = (float(intensity) - float(min_i)) / (float(max_i) - float(min_i))
    t = float(np.clip(t, 0.0, 1.0))
    return a_min + t * (a_max - a_min)

def _to_rep_distance(sim):
    sim = np.asarray(sim, dtype=float)
    return 1.0 - sim

def _get_accuracy_drop(df):
    for c in ("performance_base", "performance_base_tree", "acc_base"):
        if c in df.columns:
            return np.asarray(df[c], dtype=float) - np.asarray(df["performance_perturbed"], dtype=float)
    raise KeyError("Base-Performance column missing. Expected one of: performance_base, performance_base_tree, acc_base")

def _read_subforest_file(path):
    rows = []
    current_fold = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Leere Zeilen überspringen
            if not line:
                continue

            # Header überspringen
            if line.startswith("dataset"):
                continue

            # Normale CSV-Zeile parsen
            parts = pd.read_csv(
                pd.io.common.StringIO(line),
                header=None
            ).iloc[0]

            if parts[3] == "Combined":
                continue

            feature_importances = np.fromstring(parts[17].strip("[]"), sep=", ")

            rows.append({
                "fold": parts[2],
                "representation": parts[3],
                "selection_strategy": parts[4] if parts[4] != "" else None,
                "full_forest_size": int(parts[5]),
                "subforest_size": int(parts[6]),
                "acc": float(parts[7]),
                "macro_f1": float(parts[8]),
                "mcc": float(parts[9]),
                "roc_auc": float(parts[10]) if parts[10] != "" else None,
                "pr_auc": float(parts[11]) if parts[11] != "" else None,
                "minority_class": parts[12] if parts[12] != "" else None,
                "minority_support": float(parts[13]) if parts[13] != "" else None,
                "minority_precision": float(parts[14]) if parts[14] != "" else None,
                "minority_recall": float(parts[15]) if parts[15] != "" else None,
                "minority_f1": float(parts[16]) if parts[16] != "" else None,
                "feature_importances": feature_importances,
                "agreement_with_full_forest": float(parts[19]) if parts[19] != "" else None
            })

    df = pd.DataFrame(rows)

    return df

def _calculate_subforest_metrics(df):

    results = {
        "full_forest": df[df["representation"] == "Full Forest"]
            .groupby("subforest_size")[metrics]
            .mean()
            .reset_index(),

        "random": df[df["representation"] == "Random"]
            .groupby("subforest_size")[metrics]
            .mean()
            .reset_index(),

        "top_oob_acc": df[df["representation"] == "Top OOB ACC"]
            .groupby("subforest_size")[metrics]
            .mean()
            .reset_index(),

        "top_oob_mcc": df[df["representation"] == "Top OOB MCC"]
            .groupby("subforest_size")[metrics]
            .mean()
            .reset_index(),

        "representations": df[
            ~df["representation"].isin([
                "Full Forest",
                "Random",
                "Top OOB ACC",
                "Top OOB MCC"
            ])
        ]
        .groupby(["representation", "selection_strategy", "subforest_size"])[metrics]
        .mean()
        .reset_index()
    }

    return results

def _average_pairwise_spearman(vectors):
    if len(vectors) < 2:
        return np.nan
    corrs = []
    for i, j in combinations(range(len(vectors)), 2):
        a, b = vectors[i], vectors[j]
        if a is None or b is None or len(a) != len(b):
            continue
        corr, _ = spearmanr(a, b)
        if not np.isnan(corr):
            corrs.append(corr)
    return float(np.mean(corrs)) if corrs else np.nan

def _kendalls_w(vectors):
    if len(vectors) < 2:
        return np.nan
    n = min(len(v) for v in vectors if v is not None)
    if n < 2:
        return np.nan
    ranks = np.vstack([rankdata(v[:n], method="average") for v in vectors if v is not None])
    m = ranks.shape[0]
    R = ranks.sum(axis=0)
    R_bar = np.mean(R)
    S = np.sum((R - R_bar) ** 2)
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    return float(W)

def _calculate_feature_importance_stability(df, method):
    method = method.lower()
    rows = []

    for (rep, clustering, size), g in df.groupby(["representation", "selection_strategy", "subforest_size"], dropna=False):
        vectors = [v for v in g["feature_importances"].tolist() if v is not None]

        if method == "spearman":
            stability = _average_pairwise_spearman(vectors)
        elif method in ("kendall", "kendalls_w", "kendall_w"):
            stability = _kendalls_w(vectors)
        else:
            raise ValueError("method must be 'spearman' or 'kendall'")

        rows.append({
            "representation": rep,
            "selection_strategy": clustering,
            "subforest_size": int(size),
            "stability": stability
        })

    return pd.DataFrame(rows)
