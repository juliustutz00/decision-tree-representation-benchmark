import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timezone
import time
from src.representations.feature_graph_representation import FeatureGraphRepresentation                 # type: ignore
from src.representations.topological_forest_representation import TopologicalForestRepresentation       # type: ignore
from src.representations.indtree_representation import INDTreeRepresentation                            # type: ignore
from src.structural_difference import compute_structural_difference                                     # type: ignore
from src.perturbations import remove_nodes                                                              # type: ignore
from src.benchmark_utils import (                                                                       # type: ignore
    train_own_random_forest,
    evaluate_forest,
    prediction_agreement,
    similarity_to_distance_matrix,
    select_subforest_via_clustering,
    compute_similarity_to_base_tree,
    get_combined_correlation
)
from src.plotting import (                                                                              # type: ignore
    plot_similarity_performance, 
    plot_similarity_structural_difference
)

"""
Core benchmark orchestration.

This module runs:
1) representation perturbation benchmark, and
2) subforest selection benchmark,
including optional plotting and result persistence.
"""

def run_benchmark(
    X_train,
    X_test,
    y_train,
    y_test,
    features_info,
    representation_options,
    perturbations,
    intensities=[0.25, 0.5, 0.75, 1],
    strengths=[0],
    perturbation_runs=1,
    n_trees=100,
    subforest_size=[10],
    clustering="k-medoid",
    run_topological_forest=False,
    run_indtree=False,
    print_progress=False,
    save_results=False,
    dataset_name=None,
    results_root=r"..\results",
    representation_benchmark=True,
    subforest_selection=True,
    run_id=None,
    fold_idx=None,
    append_results=True,
    seed=0
):
    """
    Execute a full benchmark run for one dataset fold.

    Returns
    -------
    tuple[pandas.DataFrame | None, pandas.DataFrame | None]
        (perturbation_results, subforest_results)
    """
    if not representation_benchmark and not subforest_selection:
        raise ValueError(
            "At least one of representation_benchmark or subforest_selection has to be True."
        )

    if print_progress:
        print("Training random forest.")
    
    random_forest_trees, bootstrap_indices_list, oob_indices_list = (
        train_own_random_forest(X_train, y_train, n_trees, print_progress, seed)
    )

    df_perturbations = None
    df_subforest = None

    if representation_benchmark:
        df_perturbations = run_representation_benchmark(
            X_train,
            y_train,
            features_info,
            representation_options,
            perturbations,
            random_forest_trees,
            bootstrap_indices_list,
            oob_indices_list,
            intensities,
            strengths,
            perturbation_runs,
            run_topological_forest,
            run_indtree,
            print_progress,
            seed,
        )

    if subforest_selection:
        df_subforest = run_subforest_selection(
            X_train,
            X_test,
            y_train,
            y_test,
            representation_options,
            random_forest_trees,
            oob_indices_list,
            n_trees,
            subforest_size,
            clustering,
            run_topological_forest,
            run_indtree,
            print_progress,
            seed,
        )

    if representation_benchmark and df_perturbations is not None:
        if run_topological_forest:
            representation_options["Topological Forest"] = None
        if run_indtree:
            representation_options["INDTree"] = None

        if print_progress:
            print(df_perturbations.head())
            print()

        base_dir = os.path.join(results_root, "plots")
        os.makedirs(base_dir, exist_ok=True)

        rep_corr_rows = []
        for representation_name in representation_options:
            fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
            perf_path = os.path.join(
                base_dir,
                f"{representation_name.replace(' ', '_')}_similarity_performance_{seed}{fold_suffix}.png",
            )
            struct_path = os.path.join(
                base_dir,
                f"{representation_name.replace(' ', '_')}_similarity_structural_difference_{seed}{fold_suffix}.png",
            )
            try:
                _, r_perf, p_perf = plot_similarity_performance(df_perturbations, representation_name)
                _, r_struct, p_struct = plot_similarity_structural_difference(df_perturbations, representation_name)
                rep_corr_rows.append({
                    "representation": representation_name,
                    "pearson_r_performance": r_perf,
                    "pearson_p_performance": f"{p_perf:.6f}",
                    "pearson_r_structural_displayed": r_struct,
                    "pearson_p_structural": f"{p_struct:.6f}",
                })
            except Exception as e:
                if print_progress:
                    print(f"Plotting failed for {representation_name}: {e}")

        combined_corr_df = None
        try:
            combined_corr_df = get_combined_correlation(df_perturbations)
            if print_progress:
                print(combined_corr_df)
                print()
        except Exception:
            if print_progress:
                print("get_combined_correlation failed.")
                print()
        

    if subforest_selection and df_subforest is not None and print_progress:
        print(df_subforest)
        print()

    if save_results:
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        os.makedirs(results_root, exist_ok=True)
        meta = {
            "run_id": run_id,
            "fold_idx": int(fold_idx) if fold_idx is not None else None,
            "dataset_name": dataset_name,
            "n_instances_train": int(X_train.shape[0]) if X_train is not None else None,
            "n_instances_test": int(X_test.shape[0]) if X_test is not None else None,
            "n_features": int(X_train.shape[1]) if X_train is not None else None,
            "representations": list(representation_options.keys()),
            "perturbations": list(perturbations.keys()),
            "intensities": intensities,
            "perturbation_runs": perturbation_runs,
            "random_forest_size": n_trees,
            "subforest_size": subforest_size,
            "clustering": clustering,
            "run_topological_forest": run_topological_forest,
            "run_indtree": run_indtree,
            "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        }

        if representation_benchmark and df_perturbations is not None:
            pert_path = os.path.join(results_root, "perturbation_results")
            os.makedirs(pert_path, exist_ok=True)

            csv_file = os.path.join(pert_path, f"{run_id}_perturbations.csv")
            if append_results:
                _append_df_as_fold_block_csv(csv_file, df_perturbations, fold_idx)
            else:
                df_perturbations.to_csv(csv_file, index=False)

            if 'rep_corr_rows' in locals() and len(rep_corr_rows) > 0:
                corr_df = pd.DataFrame(rep_corr_rows)
                corr_file = os.path.join(pert_path, f"{run_id}_performance_ted_correlations.csv")
                if append_results:
                    _append_df_as_fold_block_csv(corr_file, corr_df, fold_idx)
                else:
                    corr_df.to_csv(corr_file, index=False)

            if 'combined_corr_df' in locals() and combined_corr_df is not None:
                comb_file = os.path.join(pert_path, f"{run_id}_representation_correlations.csv")
                if append_results:
                    _append_df_as_fold_block_csv(comb_file, combined_corr_df, fold_idx)
                else:
                    combined_corr_df.to_csv(comb_file, index=False)

            _append_jsonl(os.path.join(pert_path, f"{run_id}_metadata.jsonl"), meta)

        if subforest_selection and df_subforest is not None:
            sub_path = os.path.join(results_root, "subforest_results")
            os.makedirs(sub_path, exist_ok=True)

            csv_file = os.path.join(sub_path, f"{run_id}_subforest.csv")
            if append_results:
                _append_df_as_fold_block_csv(csv_file, df_subforest, fold_idx)
            else:
                df_subforest.to_csv(csv_file, index=False)

            _append_jsonl(os.path.join(sub_path, f"{run_id}_metadata.jsonl"), meta)

    return df_perturbations, df_subforest


def run_representation_benchmark(
    X_train,
    y_train,
    features_info,
    representation_options,
    perturbations,
    random_forest_trees,
    bootstrap_indices_list,
    oob_indices_list,
    intensities=[0.25, 0.5, 0.75, 1],
    strengths=[0.25, 0.5, 0.75, 1],
    perturbation_runs=1,
    run_topological_forest=False,
    run_indtree=False,
    print_progress=False,
    seed=0,
):
    """
    Apply perturbations to base trees and compute representation similarities.

    Output columns include perturbation metadata, base/perturbed performance,
    structural difference, and `sim_<representation>` values.
    """
    if print_progress:
        print("Running representation benchmark...")

    results = []
    topological_forest_representations = []
    topological_forest_R = None

    indtree_all_trees = []
    for idx, template_tree in enumerate(random_forest_trees):
        X_boot, y_boot = (
            X_train[bootstrap_indices_list[idx]],
            y_train[bootstrap_indices_list[idx]],
        )
        X_oob, y_oob = X_train[oob_indices_list[idx]], y_train[oob_indices_list[idx]]
        base_tree = remove_nodes(
            template_tree,
            None,
            X_boot,
            y_boot,
            features_info,
            intensity=0.5,
            strength=0,
            seed=seed,
        )
        for name, R in representation_options.items():
            if name == "Feature Graph":
                original_feature_graph = R
                new_feature_graph = FeatureGraphRepresentation(
                    criterion=original_feature_graph.criterion,
                    X=X_boot 
                )
                representation_options[name] = new_feature_graph
        if run_topological_forest:
            topological_forest_R = TopologicalForestRepresentation(tree_vectors=None)
            representation_base_tree = topological_forest_R.represent(base_tree, X_boot)
            topological_forest_representations.append(representation_base_tree)
        if run_indtree:
            indtree_all_trees.append(base_tree)

        performance_base_tree = base_tree.score(X_oob, y_oob)
        representations_base_tree = {
            name: R.represent(base_tree, X_boot)
            for name, R in representation_options.items()
        }

        for p_name, p_fn in perturbations.items():
            for i in intensities:
                for s in strengths:
                    for p_run in range(perturbation_runs):
                        # deepcopy is made in the perturbation function
                        perturbed_tree = p_fn(
                            base_tree,
                            template_tree,
                            X_boot,
                            y_boot,
                            features_info,
                            intensity=i,
                            strength=s,
                            seed=seed + idx + p_run,
                        )
                        if run_topological_forest:
                            representation_perturbed_tree = topological_forest_R.represent(
                                perturbed_tree, X_boot
                            )
                            topological_forest_representations.append(
                                representation_perturbed_tree
                            )
                        if run_indtree:
                            indtree_all_trees.append(perturbed_tree)
                        performance_perturbed_tree = perturbed_tree.score(X_oob, y_oob)
                        structural_diff = compute_structural_difference(
                            base_tree, perturbed_tree, X_boot
                        )

                        similarities = {}
                        for name, R in representation_options.items():
                            similarity = R.similarity(
                                representations_base_tree[name],
                                R.represent(perturbed_tree, X_boot),
                            )
                            similarities[name] = similarity

                        results.append(
                            {
                                "seed": int(seed + idx + p_run),
                                "perturbation": p_name,
                                "intensity": i,
                                "strength": s,
                                "performance_base": performance_base_tree,
                                "performance_perturbed": performance_perturbed_tree,
                                "structural_difference": structural_diff,
                                **{f"sim_{k}": v for k, v in similarities.items()},
                            }
                        )

    for name, R in representation_options.items():
            if name == "Feature Graph":
                original_feature_graph = R
                new_feature_graph = FeatureGraphRepresentation(
                    criterion=original_feature_graph.criterion,
                    X=X_train 
                )
                representation_options[name] = new_feature_graph
    results = pd.DataFrame(results)

    if run_topological_forest:
        if print_progress:
            print("Computing Topological Forest similarities.")

        topological_forest_R = TopologicalForestRepresentation(
            tree_vectors=topological_forest_representations
        )
        similarity_values = compute_similarity_to_base_tree(
            lambda i, j: topological_forest_R.similarity(i, j),
            len(topological_forest_representations),
            len(perturbations),
            len(intensities),
            len(strengths),
            perturbation_runs,
        )
        results["sim_Topological Forest"] = [
            sim_value["similarity_to_base"] for sim_value in similarity_values
        ]

    if run_indtree:
        if print_progress:
            print("Computing INDTree similarities.")

        # direct or repr3rows, encoding or output
        indtree_R = INDTreeRepresentation(
            indtree_all_trees, X_train, y_train, "direct", "output", seed
        )
        similarity_values = compute_similarity_to_base_tree(
            lambda i, j: indtree_R.similarity(i, j),
            len(indtree_all_trees),
            len(perturbations),
            len(intensities),
            len(strengths),
            perturbation_runs,
        )
        results["sim_INDTree"] = [
            sim_value["similarity_to_base"] for sim_value in similarity_values
        ]

    return results


def run_subforest_selection(
    X_train,
    X_test,
    y_train,
    y_test,
    representation_options,
    random_forest_trees,
    oob_indices_list=None,
    n_trees=100,
    subforest_size=[10],
    clustering="k-medoid",
    run_topological_forest=False,
    run_indtree=False,
    print_progress=False,
    seed=0,
):
    """
    Compare subforest selection strategies for multiple target subforest sizes.

    Includes baselines (full, random, top-OOB) and representation-based clustering.
    """
    if print_progress:
        print("Running subforest selection...")

    sizes = [int(s) for s in subforest_size]

    if clustering is None:
        clustering_methods = []
    elif isinstance(clustering, (list, tuple)):
        clustering_methods =  list(clustering)
    else:
        clustering_methods = [clustering]

    if len(clustering_methods) == 0:
        clustering_methods = ["k-medoid"]

    sizes = sorted({s for s in sizes if s is not None})
    if len(sizes) == 0:
        raise ValueError("subforest_size must not be empty.")
    for s in sizes:
        if s <= 0 or s > n_trees:
            raise ValueError(f"Each subforest_size must be in the range [1, {n_trees}], got {s}.")

    results = []
    n_instances_test = X_test.shape[0]
    class_labels = np.unique(y_train)
    n_classes = len(class_labels)

    # compute metrics of full random forest
    full_forest_eval = evaluate_forest(
        X_test,
        y_test,
        random_forest_trees,
        n_instances_test,
        n_classes
    )

    results.append({
        "representation": "Full Forest",
        "clustering": None,
        "full_forest_size": int(n_trees),
        "subforest_size": int(n_trees),
        "acc": full_forest_eval["metrics"]["accuracy"],
        "balanced_acc": full_forest_eval["metrics"]["balanced_accuracy"],
        "macro_f1": full_forest_eval["metrics"]["macro_f1"],
        "roc_auc": full_forest_eval["metrics"].get("roc_auc", np.nan),
    })

    # Random Subforest baseline
    rng = np.random.RandomState(seed)
    for s in sizes:
        random_subforest_indices = list(rng.choice(n_trees, size=int(s), replace=False))

        random_subforest_eval = evaluate_forest(
            X_test,
            y_test,
            [random_forest_trees[idx] for idx in random_subforest_indices],
            n_instances_test,
            n_classes
        )

        random_subforest_agreement = prediction_agreement(
            random_subforest_eval["hard_predictions"],
            full_forest_eval["hard_predictions"]
        )

        results.append({
            "representation": "Random Subforest",
            "clustering": None,
            "full_forest_size": int(n_trees),
            "subforest_size": int(s),
            "acc": random_subforest_eval["metrics"]["accuracy"],
            "balanced_acc": random_subforest_eval["metrics"]["balanced_accuracy"],
            "macro_f1": random_subforest_eval["metrics"]["macro_f1"],
            "roc_auc": random_subforest_eval["metrics"].get("roc_auc", np.nan),
            "agreement_with_full_forest": random_subforest_agreement,
            "indices": sorted([int(i) for i in random_subforest_indices]),
        })

    # Top OOB Subforest baseline
    if oob_indices_list is None or len(oob_indices_list) < len(random_forest_trees):
        raise ValueError("OOB indices list is not available for all trees.")

    oob_accs = []
    for idx, tree in enumerate(random_forest_trees):
        oob_idx = oob_indices_list[idx]
        if oob_idx is None or len(oob_idx) == 0:
            oob_accs.append(np.nan)
        else:
            try:
                acc = tree.score(X_train[oob_idx], y_train[oob_idx])
            except Exception:
                acc = np.nan
            oob_accs.append(acc)

    accs_arr = np.array([a if not np.isnan(a) else -np.inf for a in oob_accs])
    oob_ranked = list(np.argsort(-accs_arr))

    for s in sizes:
        top_oob_indices = [int(i) for i in oob_ranked[:int(s)]]

        top_oob_eval = evaluate_forest(
            X_test,
            y_test,
            [random_forest_trees[idx] for idx in top_oob_indices],
            n_instances_test,
            n_classes
        )

        top_oob_agreement = prediction_agreement(
            top_oob_eval["hard_predictions"],
            full_forest_eval["hard_predictions"]
        )

        results.append({
            "representation": "Top OOB Subforest",
            "clustering": None,
            "full_forest_size": int(n_trees),
            "subforest_size": int(s),
            "acc": top_oob_eval["metrics"]["accuracy"],
            "balanced_acc": top_oob_eval["metrics"]["balanced_accuracy"],
            "macro_f1": top_oob_eval["metrics"]["macro_f1"],
            "roc_auc": top_oob_eval["metrics"].get("roc_auc", np.nan),
            "agreement_with_full_forest": top_oob_agreement,
            "indices": sorted(top_oob_indices),
        })

    subforest_distance_matrices = []

    # representation-based subforest selection
    for name, R in representation_options.items():
        if print_progress:
            print("Starting subforest computation for representation:", name)
            start_time = time.time()

        collected_representations = [R.represent(tree, X_train) for tree in random_forest_trees]

        num_trees = len(collected_representations)
        distance_matrix = similarity_to_distance_matrix(
            lambda i, j: R.similarity(collected_representations[i], collected_representations[j]),
            num_trees,
        )
        subforest_distance_matrices.append(distance_matrix)

        for clustering_method in clustering_methods:
            for s in sizes:
                subforest_indices = select_subforest_via_clustering(
                    distance_matrix, int(s), clustering_method, seed
                )

                subforest_eval = evaluate_forest(
                    X_test,
                    y_test,
                    [random_forest_trees[idx] for idx in subforest_indices],
                    n_instances_test,
                    n_classes
                )

                subforest_agreement = prediction_agreement(
                    subforest_eval["hard_predictions"],
                    full_forest_eval["hard_predictions"]
                )

                results.append({
                    "representation": name,
                    "clustering": clustering_method,
                    "full_forest_size": int(n_trees),
                    "subforest_size": int(s),
                    "acc": subforest_eval["metrics"]["accuracy"],
                    "balanced_acc": subforest_eval["metrics"]["balanced_accuracy"],
                    "macro_f1": subforest_eval["metrics"]["macro_f1"],
                    "roc_auc": subforest_eval["metrics"].get("roc_auc", np.nan),
                    "agreement_with_full_forest": subforest_agreement,
                    "indices": sorted([int(i) for i in subforest_indices]),
                })

        if print_progress:
            end_time = time.time()
            print(f"Finished in {end_time - start_time:.2f} seconds.")

    if len(subforest_distance_matrices) != 0:
        combined_distance_matrix = np.mean(subforest_distance_matrices, axis=0)

        for clustering_method in clustering_methods:
            for s in sizes:
                subforest_indices = select_subforest_via_clustering(
                    combined_distance_matrix, int(s), clustering_method, seed
                )

                subforest_eval = evaluate_forest(
                    X_test,
                    y_test,
                    [random_forest_trees[idx] for idx in subforest_indices],
                    n_instances_test,
                    n_classes
                )

                subforest_agreement = prediction_agreement(
                    subforest_eval["hard_predictions"],
                    full_forest_eval["hard_predictions"]
                )

                results.append({
                    "representation": "Combined",
                    "clustering": clustering_method,
                    "full_forest_size": int(n_trees),
                    "subforest_size": int(s),
                    "acc": subforest_eval["metrics"]["accuracy"],
                    "balanced_acc": subforest_eval["metrics"]["balanced_accuracy"],
                    "macro_f1": subforest_eval["metrics"]["macro_f1"],
                    "roc_auc": subforest_eval["metrics"].get("roc_auc", np.nan),
                    "agreement_with_full_forest": subforest_agreement,
                    "indices": sorted([int(i) for i in subforest_indices]),
                })

    if run_topological_forest:
        if print_progress:
            print("Starting subforest computation for representation: Topological Forest")
            start_time = time.time()

        topological_forest_R = TopologicalForestRepresentation(tree_vectors=None)
        topological_forest_representations = [
            topological_forest_R.represent(t, X_train) for t in random_forest_trees
        ]
        topological_forest_R = TopologicalForestRepresentation(
            tree_vectors=topological_forest_representations
        )

        num_trees = len(topological_forest_representations)
        distance_matrix = similarity_to_distance_matrix(
            lambda i, j: topological_forest_R.similarity(i, j),
            num_trees
        )

        for clustering_method in clustering_methods:
            for s in sizes:
                subforest_indices = select_subforest_via_clustering(
                    distance_matrix, int(s), clustering_method, seed
                )

                subforest_eval = evaluate_forest(
                    X_test,
                    y_test,
                    [random_forest_trees[idx] for idx in subforest_indices],
                    n_instances_test,
                    n_classes
                )
                
                subforest_agreement = prediction_agreement(
                    subforest_eval["hard_predictions"],
                    full_forest_eval["hard_predictions"]
                )

                results.append({
                    "representation": "Topological Forest",
                    "clustering": clustering_method,
                    "full_forest_size": int(n_trees),
                    "subforest_size": int(s),
                    "acc": subforest_eval["metrics"]["accuracy"],
                    "balanced_acc": subforest_eval["metrics"]["balanced_accuracy"],
                    "macro_f1": subforest_eval["metrics"]["macro_f1"],
                    "roc_auc": subforest_eval["metrics"].get("roc_auc", np.nan),
                    "agreement_with_full_forest": subforest_agreement,
                    "indices": sorted([int(i) for i in subforest_indices]),
                })

        if print_progress:
            end_time = time.time()
            print(f"Topological Forest finished in {end_time - start_time:.2f} seconds.")

    if run_indtree:
        if print_progress:
            print("Starting subforest computation for representation: INDTree")
            start_time = time.time()

        ind_R = INDTreeRepresentation(
            random_forest_trees, X_train, y_train, "direct", "output", seed
        )

        num_trees = len(random_forest_trees)
        distance_matrix = similarity_to_distance_matrix(
            lambda i, j: ind_R.similarity(i, j),
            num_trees
        )

        for clustering_method in clustering_methods:
            for s in sizes:
                subforest_indices = select_subforest_via_clustering(
                    distance_matrix, int(s), clustering_method, seed
                )

                subforest_eval = evaluate_forest(
                    X_test,
                    y_test,
                    [random_forest_trees[idx] for idx in subforest_indices],
                    n_instances_test,
                    n_classes
                )
                
                subforest_agreement = prediction_agreement(
                    subforest_eval["hard_predictions"],
                    full_forest_eval["hard_predictions"]
                )

                results.append({
                    "representation": "INDTree",
                    "clustering": clustering_method,
                    "full_forest_size": int(n_trees),
                    "subforest_size": int(s),
                    "acc": subforest_eval["metrics"]["accuracy"],
                    "balanced_acc": subforest_eval["metrics"]["balanced_accuracy"],
                    "macro_f1": subforest_eval["metrics"]["macro_f1"],
                    "roc_auc": subforest_eval["metrics"].get("roc_auc", np.nan),
                    "agreement_with_full_forest": subforest_agreement,
                    "indices": sorted([int(i) for i in subforest_indices]),
                })

        if print_progress:
            end_time = time.time()
            print(f"INDTree finished in {end_time - start_time:.2f} seconds.")

    results = pd.DataFrame(results)
    return results


def _append_df_as_fold_block_csv(path, df, fold_idx):
    """Append a dataframe to CSV; optionally prepend a `# Fold <idx>` marker."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.exists(path)

    with open(path, "a", encoding="utf-8", newline="") as fh:
        if not file_exists:
            df.head(0).to_csv(fh, index=False)

        if fold_idx is not None:
            fh.write(f"# Fold {int(fold_idx)}\n")

        if len(df) > 0:
            df.to_csv(fh, index=False, header=False)

def _append_jsonl(path, obj):
    """Append one JSON object as a single line to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
