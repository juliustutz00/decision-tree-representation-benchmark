import copy
from pathlib import Path
import yaml
from datetime import datetime, timezone
import pandas as pd
import os

from src.benchmark import run_benchmark # type: ignore
from src.plotting import (  # type: ignore
    plot_similarity_performance,
    plot_similarity_structural_difference,
)

from src.data_utils import (    # type: ignore
    get_iris_dataset,
    get_breast_cancer_dataset,
    get_wine_dataset,
    get_MAGIC_gamma_dataset,
    get_kidney_dataset,
    get_AI4I_dataset,
    get_adult_dataset,
    get_letter_recognition_dataset,
    get_bank_marketing_dataset,
    get_banknote_authentication_dataset,
    get_aml_TCGA_dataset,
    get_bic_TCGA_dataset,
    get_coad_TCGA_dataset,
    get_gbm_TCGA_dataset,
    get_kirc_TCGA_dataset,
    get_lihc_TCGA_dataset,
    get_lusc_TCGA_dataset,
    get_skcm_TCGA_dataset,
    get_ov_TCGA_dataset,
    get_sarc_TCGA_dataset,
)

from src.perturbations import ( # type: ignore
    change_threshold,
    change_feature,
    swap_nodes,
    remove_nodes,
    add_nodes,
)

from src.representations.feature_graph_representation import FeatureGraphRepresentation # type: ignore
from src.representations.tree_descriptor_representation import TreeDescriptorRepresentation  # type: ignore
from src.representations.leaf_profile_representation import LeafProfileRepresentation   # type: ignore


DATASETS = {
    "iris": get_iris_dataset,
    "breast_cancer": get_breast_cancer_dataset,
    "wine": get_wine_dataset,
    "MAGIC_gamma": get_MAGIC_gamma_dataset,
    "kidney": get_kidney_dataset,
    "AI4I": get_AI4I_dataset,
    "adult": get_adult_dataset,
    "letter_recognition": get_letter_recognition_dataset,
    "bank_marketing": get_bank_marketing_dataset,
    "banknote_authentication": get_banknote_authentication_dataset,
    "aml": get_aml_TCGA_dataset,
    "bic": get_bic_TCGA_dataset,
    "coad": get_coad_TCGA_dataset,
    "gbm": get_gbm_TCGA_dataset,
    "kirc": get_kirc_TCGA_dataset,
    "lihc": get_lihc_TCGA_dataset,
    "lusc": get_lusc_TCGA_dataset,
    "skcm": get_skcm_TCGA_dataset,
    "ov": get_ov_TCGA_dataset,
    "sarc": get_sarc_TCGA_dataset,
}

REPRESENTATION_BUILDERS = {
    "Feature Graph": lambda X_train, _: FeatureGraphRepresentation(criterion="sample", X=X_train),
    "Tree Descriptor": lambda _, __: TreeDescriptorRepresentation(weights=None, metric="cosine"),
    "Leaf Profile": lambda _, __: LeafProfileRepresentation(criterion="l2")
}

PERTURBATIONS = {
    "change_threshold": change_threshold,
    "change_feature": change_feature,
    "swap_nodes": swap_nodes,
    "remove_nodes": remove_nodes,
    "add_nodes": add_nodes,
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a deep copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_representation_options(cfg: dict, X_train) -> dict:
    """Instantiate enabled representations for the current fold."""
    flags = cfg.get("representations", {}) or {}
    opts = {}
    for name, enabled in flags.items():
        if not enabled:
            continue
        builder = REPRESENTATION_BUILDERS.get(name)
        if builder is None:
            raise KeyError(f"No builder defined for representation '{name}'")
        opts[name] = builder(X_train, cfg.get("seed", 0))
    return opts


def build_perturbations(cfg: dict) -> dict:
    """Build dictionary of enabled perturbation callables from config flags."""
    flags = cfg.get("perturbations", {}) or {}
    perts = {}
    for name, enabled in flags.items():
        if enabled:
            perts[name] = PERTURBATIONS[name]
    return perts


def main(config_path: str = str(Path(__file__).with_name("benchmark_runs.yaml"))):
    """Run all configured benchmark runs and folds from a YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {}) or {}
    runs = raw.get("runs", []) or []

    for run_idx, run_override in enumerate(runs, start=1):
        cfg = deep_merge(defaults, run_override)

        dataset_name = cfg["dataset"]
        loader = DATASETS[dataset_name]
        (folds, features_info), resolved_name = loader(n_splits=cfg.get("n_splits", 3), n_samples=cfg.get("n_samples", 10000), seed=cfg.get("seed", 0))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_run{run_idx}_{dataset_name}"

        pooled_perturbations = []

        for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
            fold_seed = cfg.get("seed", 0) + fold_idx
            
            representation_options = build_representation_options(cfg, X_train)
            perturbations = build_perturbations(cfg)

            df_perturbations, df_subforest = run_benchmark(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                features_info=features_info,
                representation_options=representation_options,
                perturbations=perturbations,
                intensities=cfg.get("intensities", [0.2, 0.4, 0.6, 0.8, 1]),
                strengths=cfg.get("strengths", [0]),
                perturbation_runs=cfg.get("perturbation_runs", 1),
                n_trees=cfg.get("random_forest_size", 100),
                subforest_size=cfg.get("subforest_size", [5, 10, 15, 20]),
                clustering=cfg.get("clustering", "k-medoid"),
                run_topological_forest=cfg.get("run_topological_forest", False),
                run_indtree=cfg.get("run_indtree", False),
                print_progress=cfg.get("print_progress", False),
                save_results=cfg.get("save_results", False),
                dataset_name=f"{resolved_name}_fold{fold_idx}",
                results_root=cfg.get("results_root", r"..\results"),
                representation_benchmark=cfg.get("representation_benchmark", True),
                subforest_selection=cfg.get("subforest_selection", True),
                run_id=run_id,
                fold_idx=fold_idx,
                append_results=True,
                seed=fold_seed,
            )

            if df_perturbations is not None and len(df_perturbations) > 0:
                df_perturbations = df_perturbations.copy()
                df_perturbations["fold_idx"] = fold_idx + 1
                pooled_perturbations.append(df_perturbations)
            
            print(f"Finished run {run_idx}/{len(runs)} | "f"dataset={dataset_name} | fold={fold_idx+1}/{len(folds)}")

        if cfg.get("save_results", False) and len(pooled_perturbations) > 0:
            pooled_df = pd.concat(pooled_perturbations, ignore_index=True)

            results_root = cfg.get("results_root", r"..\results")
            plots_dir = os.path.join(results_root, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            reps = [c[len("sim_"):] for c in pooled_df.columns if c.startswith("sim_")]

            for rep in reps:
                perf_path = os.path.join(plots_dir, f"{run_id}_{rep.replace(' ', '_')}_similarity_performance_ALLFOLDS.png")
                struct_path = os.path.join(plots_dir, f"{run_id}_{rep.replace(' ', '_')}_similarity_structural_difference_ALLFOLDS.png")

                plot_similarity_performance(pooled_df, rep, save_results=True, savepath=perf_path, fold_col="fold_idx", color_by_fold=True)
                plot_similarity_structural_difference(pooled_df, rep, save_results=True, savepath=struct_path, fold_col="fold_idx", color_by_fold=True)


if __name__ == "__main__":
    main()
