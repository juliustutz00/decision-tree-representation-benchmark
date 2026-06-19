import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation
from dataclasses import dataclass
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import (
    matthews_corrcoef,
    silhouette_score,
)

MIN_SUBFOREST_SIZE = None
_ZSCALE_N_RANDOM_SOLUTIONS = 100

# main method for selecting a subforest based on clustering of the distance matrix, with multiple clustering method options
def select_subforest_via_clustering(distance_matrix, subforest_size, clustering_method, seed, random_forest_trees=None,
    X_train=None, y_train=None, oob_indices_list=None, density_sigma_grid=None, density_alpha_grid=None, name=None, size=None, savepath=None):
    global MIN_SUBFOREST_SIZE
    _validate_subforest_size(subforest_size, distance_matrix.shape[0])
    clustering_silhouette_score = None
    if clustering_method in ["density", "k-medoid-performance", "agglomerative-performance", "combination-greedy", "combination-simulated_annealing", "combination-genetic"]:
        n_classes = len(np.unique(y_train))
        all_oob_preds = _precompute_all_oob_predictions(random_forest_trees, oob_indices_list, X_train)
    
    if clustering_method == "k-medoid":
        kmed_clustering = _cluster_labels(distance_matrix, subforest_size, method="k-medoid", seed=seed)
        subforest_indices = [int(x) for x in kmed_clustering.medoid_indices_]
        clustering_silhouette_score = silhouette_score(distance_matrix, kmed_clustering.labels_, metric="precomputed")
    elif clustering_method == "agglomerative":
        labels = _cluster_labels(distance_matrix, subforest_size, method="agglomerative", seed=seed)
        subforest_indices = []
        for cluster_id in range(subforest_size):
            cluster_indices = np.where(labels == cluster_id)[0]
            if cluster_indices.size == 0:
                continue
            cluster_distances = distance_matrix[
                np.ix_(cluster_indices, cluster_indices)
            ]
            medoid_index_within_cluster = cluster_indices[
                int(np.argmin(cluster_distances.sum(axis=1)))
            ]
            subforest_indices.append(medoid_index_within_cluster)
        clustering_silhouette_score = silhouette_score(distance_matrix, labels, metric="precomputed")
    elif clustering_method == "redundancy":
        subforest_indices = _select_subforest_via_redundancy(distance_matrix, subforest_size)
    elif clustering_method == "density":
        subforest_indices = _select_subforest_via_density(distance_matrix, subforest_size, seed, all_oob_preds, y_train=y_train, n_classes=n_classes,
                                oob_indices_list=oob_indices_list, sigma_grid=density_sigma_grid, alpha_grid=density_alpha_grid, name=name, size=size)
    elif clustering_method == "k-medoid-performance":
        subforest_indices, clustering_silhouette_score = _select_subforest_via_performance_clustering(distance_matrix, subforest_size, method="k-medoid", all_oob_preds=all_oob_preds,
                                                                        oob_indices_list=oob_indices_list, y_train=y_train, n_classes=n_classes, seed=seed)
    elif clustering_method == "agglomerative-performance":
        subforest_indices, clustering_silhouette_score = _select_subforest_via_performance_clustering(distance_matrix, subforest_size, method="agglomerative", all_oob_preds=all_oob_preds,
                                                                        oob_indices_list=oob_indices_list, y_train=y_train, n_classes=n_classes, seed=seed)
    elif clustering_method == "combination-greedy":
        subforest_indices = _select_subforest_via_combination(distance_matrix, subforest_size, method="greedy", all_oob_preds=all_oob_preds,
                                                              oob_indices_list=oob_indices_list, y_train=y_train, n_classes=n_classes, seed=seed)
    elif clustering_method == "combination-simulated_annealing":
        subforest_indices = _select_subforest_via_combination(distance_matrix, subforest_size, method="simulated_annealing", all_oob_preds=all_oob_preds,
                                                                oob_indices_list=oob_indices_list, y_train=y_train, n_classes=n_classes, seed=seed)
    elif clustering_method == "combination-genetic":
        subforest_indices = _select_subforest_via_combination(distance_matrix, subforest_size, method="genetic", all_oob_preds=all_oob_preds,
                                                                oob_indices_list=oob_indices_list, y_train=y_train, n_classes=n_classes, seed=seed)
    elif clustering_method == "k-medoid-silhouette":
        if MIN_SUBFOREST_SIZE is None or subforest_size <= MIN_SUBFOREST_SIZE:
            MIN_SUBFOREST_SIZE = subforest_size
            ks, scores = _compute_silhouette_curve(distance_matrix, method="k-medoid", seed=seed)
            _save_silhouette_curve_plot(
                ks, scores,
                method_name="k-medoid",
                save_dir=savepath,
                name=name,
                size=size
            )
        rng = np.random.RandomState(seed)
        n_trees = distance_matrix.shape[0]
        pick = subforest_size
        subforest_indices = rng.choice(np.arange(n_trees), size=pick, replace=False).astype(int).tolist()
    elif clustering_method == "agglomerative-silhouette":
        if MIN_SUBFOREST_SIZE is None or subforest_size <= MIN_SUBFOREST_SIZE:
            MIN_SUBFOREST_SIZE = subforest_size
            ks, scores = _compute_silhouette_curve(distance_matrix, method="agglomerative", seed=seed)
            _save_silhouette_curve_plot(
                ks, scores,
                method_name="agglomerative",
                save_dir=savepath,
                name=name,
                size=size
            )
        rng = np.random.RandomState(seed)
        n_trees = distance_matrix.shape[0]
        pick = subforest_size
        subforest_indices = rng.choice(np.arange(n_trees), size=pick, replace=False).astype(int).tolist()
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    
    return subforest_indices, clustering_silhouette_score

# single selection strategies
def _select_subforest_via_redundancy(distance_matrix, subforest_size):
    n_trees = distance_matrix.shape[0]

    num_remove = n_trees - subforest_size

    subforest_indices = _iteratively_remove_most_redundant(
        distance_matrix, num_remove
    )

    return subforest_indices

def _select_subforest_via_density(distance_matrix, subforest_size, seed, all_oob_preds, y_train, n_classes, oob_indices_list=None, 
                                 sigma_grid=None, alpha_grid=None, visualize=False, gif_path=r"F:\SUBFOREST\results\HPC_results\mcc_results\density_selection", 
                                 gif_fps=1, gif_step_pause_sec=2.0, gif_last_pause_sec=3.0, name=None, size=None):
    # sigma = "Einflussradius"; bei kleinem sigma zählen nur sehr nahe Nachbarn für die Dichte, bei großem sigma zählen auch weiter entfernte
    # the bigger alpha, the more the probabilities of remaining trees are reduced after selecting a tree with many close neighbors
    # future work: try to compute density using parzen windows
    gif_path = gif_path + "_" + name + "_" + str(size) + ".gif" if gif_path else None

    sigma_candidates = _default_sigma_grid(distance_matrix) if sigma_grid is None else np.asarray(sigma_grid, dtype=float)
    sigma_candidates = sigma_candidates[np.isfinite(sigma_candidates) & (sigma_candidates > 0)]
    if sigma_candidates.size == 0:
        sigma_candidates = np.array([1e-8], dtype=float)

    alpha_candidates = _default_alpha_grid() if alpha_grid is None else np.asarray(alpha_grid, dtype=float)
    alpha_candidates = alpha_candidates[np.isfinite(alpha_candidates) & (alpha_candidates > 0)]
    if alpha_candidates.size == 0:
        alpha_candidates = np.array([1e-8], dtype=float)

    best_sigma = None
    best_alpha = None
    best_mcc = -np.inf
    best_sel = None

    trial = 0
    for s in sigma_candidates:
        for a in alpha_candidates:
            sel = _select_subforest_density_once(distance_matrix, subforest_size, seed + trial, float(s), float(a))
            trial += 1
            mcc = _subforest_oob_mcc(sel, all_oob_preds, oob_indices_list, y_train, n_classes)
            if np.isfinite(mcc) and mcc > best_mcc:
                best_mcc = mcc
                best_sigma = float(s)
                best_alpha = float(a)
                best_sel = sel

    print(f"[Density-Selection] Best sigma={best_sigma:.6g}, alpha={best_alpha:.6g}, OOB-MCC={best_mcc:.6f}")

    if visualize and gif_path:
        _, best_trace = _select_subforest_density_once(distance_matrix, subforest_size, seed, best_sigma, best_alpha, return_trace=True)
        _save_density_selection_gif(distance_matrix, best_trace, gif_path, seed=seed, fps=gif_fps, step_pause_sec=gif_step_pause_sec, last_pause_sec=gif_last_pause_sec)

    return best_sel

def _select_subforest_via_performance_clustering(distance_matrix, subforest_size, method, all_oob_preds, oob_indices_list, y_train, n_classes, seed):
    mcc_per_tree = np.array([_subforest_oob_mcc([i], all_oob_preds, oob_indices_list, y_train, n_classes)for i in range(distance_matrix.shape[0])])
    
    if method == "k-medoid":
        kmed_clustering = _cluster_labels(distance_matrix, subforest_size, method="k-medoid", seed=seed)
        cluster_labels = kmed_clustering.labels_
        clustering_silhouette_score = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
    elif method == "agglomerative":
        cluster_labels = _cluster_labels(distance_matrix, subforest_size, method="agglomerative", seed=seed)
        clustering_silhouette_score = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # select best performing tree from each cluster
    subforest_indices = []
    for cluster_id in range(subforest_size):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if cluster_indices.size == 0:
            continue
        
        # get OOB MCC for each tree in cluster
        cluster_performances = mcc_per_tree[cluster_indices]
        
        # select tree with highest performance
        valid_mask = ~np.isnan(cluster_performances)
        if np.any(valid_mask):
            best_local_idx = np.nanargmax(cluster_performances)
        else:
            best_local_idx = 0
        
        subforest_indices.append(int(cluster_indices[best_local_idx]))
    
    return subforest_indices, clustering_silhouette_score

def _select_subforest_via_combination(distance_matrix, subforest_size, method, all_oob_preds, oob_indices_list, y_train, n_classes, seed):
    mcc_computation = "per_tree"

    params = {
        "distance_matrix": distance_matrix,
        "subforest_size": subforest_size,
        "all_oob_preds": all_oob_preds,
        "oob_indices_list": oob_indices_list,
        "y_train": y_train,
        "n_classes": n_classes,
        "mcc_computation": mcc_computation,
        "seed": seed
    }

    if method == "greedy":
        return _select_subforest_via_combination_greedy(**params)
    elif method == "simulated_annealing":
        return _select_subforest_via_combination_simulated_annealing(**params)
    elif method == "genetic":
        return _select_subforest_via_combination_genetic(**params)
    else:
        raise ValueError(f"Unknown combination method: {method}")

def _select_subforest_via_combination_greedy(distance_matrix, subforest_size, all_oob_preds, oob_indices_list, y_train, n_classes, mcc_computation, seed):
    # Option "per_tree": MCC sum uses per-tree MCC
    # Option "subforest": MCC sum uses subforest MCC after adding candidate (slow)
    rng = np.random.RandomState(seed)
    n_trees = distance_matrix.shape[0]
    w_div = 0.5
    w_perf = 0.5

    mcc_per_tree = np.array([_subforest_oob_mcc([i], all_oob_preds, oob_indices_list, y_train, n_classes)for i in range(n_trees)])
    if np.all(~np.isfinite(mcc_per_tree)):
        mcc_per_tree = np.zeros(n_trees, dtype=float)

    # scaling
    mean_dist_to_all = distance_matrix.mean(axis=1)
    dist_scaler = _fit_zscaler(mean_dist_to_all)
    
    if mcc_computation == "subforest":
        _, perf_scaler = _fit_combination_zscalers_from_random_solutions(
            distance_matrix=distance_matrix, subforest_size=subforest_size, mcc_per_tree=mcc_per_tree, 
            all_oob_preds=all_oob_preds, oob_indices_list=oob_indices_list, y_train=y_train, n_classes=n_classes,
            mcc_computation="subforest", seed=seed
        )
    else:
        perf_scaler = _fit_zscaler(mcc_per_tree)

    selected = []
    remaining = set(range(n_trees))

    # first tree
    z_div0 = -dist_scaler.transform(mean_dist_to_all)
    z_perf0 = perf_scaler.transform(mcc_per_tree)
    score0 = (w_div * z_div0 + w_perf * z_perf0) / (w_div + w_perf)

    first = int(rng.choice(np.where(score0 == np.nanmax(score0))[0]))
    selected.append(first)
    remaining.remove(first)

    # subsequent trees
    while len(selected) < subforest_size:
        rem = np.array(list(remaining), dtype=int)
        
        # diversity
        avg_dist_to_selected = distance_matrix[np.ix_(rem, selected)].mean(axis=1)
        z_div = dist_scaler.transform(avg_dist_to_selected)

        # performance
        if mcc_computation == "subforest":
            perf_vals = []
            for cand in rem:
                mcc = _subforest_oob_mcc(selected + [int(cand)], all_oob_preds, oob_indices_list, y_train, n_classes)
                perf_vals.append(mcc)
            perf_vals = np.array(perf_vals, dtype=float)
        else:
            k = len(selected)
            sel_sum = np.nansum(mcc_per_tree[selected])
            perf_vals = (sel_sum + mcc_per_tree[rem]) / (k + 1)
            
        z_perf = perf_scaler.transform(perf_vals)
        scores = (w_div * z_div + w_perf * z_perf) / (w_div + w_perf)

        best_idx = rem[np.where(scores == np.nanmax(scores))[0]]
        chosen = int(rng.choice(best_idx))
        selected.append(chosen)
        remaining.remove(chosen)

    return selected

def _select_subforest_via_combination_simulated_annealing(distance_matrix, subforest_size, all_oob_preds, oob_indices_list, y_train, n_classes, mcc_computation, seed):
    # Option "per_tree": MCC sum uses per-tree MCC
    # Option "subforest": MCC sum uses subforest MCC after adding candidate (slow)
    rng = np.random.RandomState(seed)
    n_trees = distance_matrix.shape[0]
    w_div = 0.5
    w_perf = 0.5

    # per-tree MCC
    mcc_per_tree = np.array([_subforest_oob_mcc([i], all_oob_preds, oob_indices_list, y_train, n_classes)for i in range(n_trees)])
    if np.all(~np.isfinite(mcc_per_tree)):
        mcc_per_tree = np.zeros(n_trees, dtype=float)

    # scaling 
    dist_scaler, perf_scaler = _fit_combination_zscalers_from_random_solutions(distance_matrix=distance_matrix, subforest_size=subforest_size, mcc_per_tree=mcc_per_tree, 
            all_oob_preds=all_oob_preds, oob_indices_list=oob_indices_list, y_train=y_train, n_classes=n_classes,
            mcc_computation="subforest", seed=seed)

    best_solution_trees = current_solution_trees = rng.choice(n_trees, size=subforest_size, replace=False).tolist()
    best_solution_score = current_solution_score = _score_solution(current_solution_trees, distance_matrix, mcc_per_tree, w_div, w_perf, dist_scaler, perf_scaler, all_oob_preds, 
                                                                   oob_indices_list, y_train, n_classes, mcc_computation)

    # hyperparameter testing
    average_delta_score = []
    for _ in range(25): 
        candidate_solution_trees = current_solution_trees.copy()
        idx_to_replace = rng.choice(subforest_size)
        available_indices = list(set(range(n_trees)) - set(candidate_solution_trees))
        
        candidate_solution_trees[idx_to_replace] = rng.choice(available_indices)
        candidate_solution_score = _score_solution(candidate_solution_trees, distance_matrix, mcc_per_tree, w_div, w_perf, dist_scaler, perf_scaler, 
                                                   all_oob_preds, oob_indices_list, y_train, n_classes, mcc_computation)
        
        average_delta_score.append(candidate_solution_score - current_solution_score)
    average_delta_score = float(np.nanmean(average_delta_score))


    # hyperparameters
    if not np.isfinite(average_delta_score) or abs(average_delta_score) < 1e-12: 
        current_temp = 1.0
    else:
        current_temp = average_delta_score / np.log(0.8)
        if not np.isfinite(current_temp) or current_temp <= 0:
            current_temp = abs(float(current_temp)) if np.isfinite(current_temp) else 1.0
            current_temp = max(current_temp, 1e-3)
    min_temp = 0.1
    cooling_rate = 0.98
    steps_per_temp = n_trees

    while current_temp > min_temp:
        for _ in range(steps_per_temp):
            candidate_solution_trees = current_solution_trees.copy()
            idx_to_replace = rng.choice(subforest_size)
            available_indices = list(set(range(n_trees)) - set(candidate_solution_trees))
            candidate_solution_trees[idx_to_replace] = rng.choice(available_indices)
            candidate_solution_score = _score_solution(candidate_solution_trees, distance_matrix, mcc_per_tree, w_div, w_perf, dist_scaler, perf_scaler, 
                                                       all_oob_preds, oob_indices_list, y_train, n_classes, mcc_computation)
            
            if candidate_solution_score > best_solution_score:
                best_solution_trees = candidate_solution_trees.copy()
                best_solution_score = candidate_solution_score

            if candidate_solution_score > current_solution_score:
                current_solution_trees = candidate_solution_trees.copy()
                current_solution_score = candidate_solution_score
            else:
                score_diff = candidate_solution_score - current_solution_score
                acceptance_prob = np.exp(score_diff / current_temp) if score_diff < 0 else 1.0
                if rng.rand() < acceptance_prob:
                    current_solution_trees = candidate_solution_trees.copy()
                    current_solution_score = candidate_solution_score
        
        current_temp *= cooling_rate

    return best_solution_trees

def _select_subforest_via_combination_genetic(distance_matrix, subforest_size, all_oob_preds, oob_indices_list, y_train, n_classes, mcc_computation, seed):
    # Option "per_tree": MCC sum uses per-tree MCC
    # Option "subforest": MCC sum uses subforest MCC after adding candidate (slow)
    rng = np.random.RandomState(seed)
    n_trees = distance_matrix.shape[0]
    w_div = 0.5
    w_perf = 0.5

    if subforest_size <= 0 or subforest_size > n_trees:
        raise ValueError(f"subforest_size must be in the range [1, {n_trees}]")

    # hyperparameters
    population_size = 100
    elite_percentage = 0.03
    crossover_probability = 0.9
    mutation_probability = (1 / (subforest_size * 2))
    max_generations = 50

    # per-tree MCC
    mcc_per_tree = np.array([_subforest_oob_mcc([i], all_oob_preds, oob_indices_list, y_train, n_classes)for i in range(n_trees)])
    if np.all(~np.isfinite(mcc_per_tree)):
        mcc_per_tree = np.zeros(n_trees, dtype=float)

    # scaling
    dist_scaler, perf_scaler = _fit_combination_zscalers_from_random_solutions(distance_matrix=distance_matrix, subforest_size=subforest_size, mcc_per_tree=mcc_per_tree, 
                                                                               all_oob_preds=all_oob_preds, oob_indices_list=oob_indices_list, y_train=y_train, n_classes=n_classes,
                                                                               mcc_computation=mcc_computation, seed=seed, n_random_solutions=_ZSCALE_N_RANDOM_SOLUTIONS)

    # initial population
    population = [
        rng.choice(n_trees, size=subforest_size, replace=False).tolist()
        for _ in range(population_size)
    ]

    best_solution = None
    best_score = -np.inf

    for _ in range(max_generations):
        scores = np.array([
            _score_solution(sol, distance_matrix, mcc_per_tree, w_div, w_perf, dist_scaler, perf_scaler, all_oob_preds, oob_indices_list, 
                            y_train, n_classes, mcc_computation) for sol in population], dtype=float)

        # track best
        gen_best_idx = int(np.nanargmax(scores))
        if scores[gen_best_idx] > best_score:
            best_score = float(scores[gen_best_idx])
            best_solution = population[gen_best_idx].copy()

        # elites
        elite_count = max(1, int(round(population_size * elite_percentage)))
        elite_indices = np.argsort(scores)[-elite_count:]
        next_generation = [population[i].copy() for i in elite_indices]

        # fill rest of generation
        while len(next_generation) < population_size:
            parent_indices = _sus_select(scores, 2, rng) 
            p1 = population[parent_indices[0]]
            p2 = population[parent_indices[1]]

            if rng.rand() < crossover_probability and subforest_size > 1:
                cut = rng.randint(1, subforest_size)
                c1 = _crossover(p1, p2, cut)
                c2 = _crossover(p2, p1, cut)
                c1 = _mutate(c1, mutation_probability, n_trees, rng)
                c2 = _mutate(c2, mutation_probability, n_trees, rng)
                next_generation.append(c1)
                if len(next_generation) < population_size:
                    next_generation.append(c2)
            else:
                c1 = _mutate(p1.copy(), mutation_probability, n_trees, rng)
                c2 = _mutate(p2.copy(), mutation_probability, n_trees, rng)
                next_generation.append(c1)
                if len(next_generation) < population_size:
                    next_generation.append(c2)

        population = next_generation

    return best_solution

# clustering-based selection helper functions
def _cluster_labels(distance_matrix, subforest_size, method, seed):
    if method == "k-medoid":
        model = KMedoids(n_clusters=int(subforest_size), metric="precomputed", random_state=seed)
        model.fit(distance_matrix)
        return model
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=int(subforest_size), metric="precomputed", linkage="average")
        return model.fit_predict(distance_matrix)
    raise ValueError(f"Unknown clustering method: {method}")

# redundancy-based selection helper functions
def _iteratively_remove_most_redundant(D, num_remove):
    n = D.shape[0]
    remaining_indices = list(range(n))
    
    for _ in range(num_remove):
        D_sub = D[np.ix_(remaining_indices, remaining_indices)]
        scores = _compute_redundancy_scores(D_sub)
        to_remove_local = np.argmin(scores)
        to_remove_global = remaining_indices[to_remove_local]
        remaining_indices.remove(to_remove_global)
    
    return remaining_indices

def _compute_redundancy_scores(D):
    return D.mean(axis=1)

# density-based selection helper functions
def _select_subforest_density_once(distance_matrix, subforest_size, seed, sigma, alpha, return_trace=False):
    rng = np.random.RandomState(seed)
    n_trees = distance_matrix.shape[0]
    densities = np.exp(-(distance_matrix ** 2) / max(float(sigma), 1e-12)).sum(axis=1)
    probs = densities / max(float(densities.sum()), 1e-12)

    remaining = set(range(n_trees))
    selected = []
    trace = []

    for step in range(subforest_size):
        rem = list(remaining)

        if return_trace:
            remaining_mask = np.zeros(n_trees, dtype=bool)
            remaining_mask[rem] = True

        p = np.array([probs[i] for i in rem], dtype=float)
        p_sum = p.sum()
        if p_sum <= 0 or not np.isfinite(p_sum):
            p = np.ones_like(p) / len(p)
        else:
            p = p / p_sum
        chosen = int(rng.choice(rem, p=p))
        selected.append(chosen)
        remaining.remove(chosen)

        if return_trace:
            trace.append({
                "step": step + 1,
                "probs": probs.copy(),
                "selected_so_far": selected.copy(),
                "chosen": chosen,
                "remaining_mask": remaining_mask,
            })

        for i in remaining:
            probs[i] *= distance_matrix[i, chosen] ** alpha

    if return_trace:
        return selected, trace
    return selected

def _default_sigma_grid(distance_matrix):
    upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    if upper.size == 0:
        return np.array([1.0], dtype=float)
    med = float(np.median(upper)) if np.median(upper) > 0 else 1e-12
    std = float(np.std(upper))
    k = std + 3
    return np.geomspace(med/k, med*k, 17)

def _default_alpha_grid():
    return np.geomspace(1/3, 3.0, 17, dtype=float)

def _save_density_selection_gif(distance_matrix, trace, savepath, seed=0, fps=1, step_pause_sec=5.0, last_pause_sec=5.0):
    if trace is None or len(trace) == 0:
        return

    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    base_fps = max(1, int(fps))
    step_repeat = max(1, int(round(step_pause_sec * base_fps)))
    last_repeat = max(1, int(round(last_pause_sec * base_fps)))

    expanded_trace = []
    for fr in trace:
        expanded_trace.extend([fr] * step_repeat)
    expanded_trace.extend([trace[-1]] * last_repeat)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, normalized_stress="auto", n_init=4)
    coords = mds.fit_transform(distance_matrix)
    x, y = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(7, 5))

    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="relative selection probability (remaining)")

    def _draw(frame_idx):
        ax.clear()
        frame = expanded_trace[frame_idx]
        probs = frame["probs"]
        selected_so_far = frame["selected_so_far"]
        chosen = frame["chosen"]
        remaining_mask = frame["remaining_mask"]

        ax.scatter(x, y, c="lightgray", s=40, alpha=0.5, edgecolors="none")

        rem_idx = np.where(remaining_mask)[0]
        if rem_idx.size > 0:
            p = probs[rem_idx]
            p_norm = (p - p.min()) / (p.max() - p.min() + 1e-12)
            ax.scatter(
                x[rem_idx], y[rem_idx], c=p_norm, cmap="viridis", vmin=0.0, vmax=1.0,
                s=60, edgecolors="black", linewidth=0.3
            )

        if len(selected_so_far) > 0:
            sel = np.array(selected_so_far, dtype=int)
            ax.scatter(x[sel], y[sel], c="dodgerblue", s=80, edgecolors="black", linewidth=0.4, label="selected so far")

        if chosen is not None:
            ax.scatter([x[chosen]], [y[chosen]], c="red", s=180, marker="*", edgecolors="black", linewidth=0.6, label="chosen now")

        ax.set_title(f"Density-based selection (step {frame['step']})")
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(loc="best", frameon=False, fontsize=8)

    ani = animation.FuncAnimation(
        fig, _draw, frames=len(expanded_trace), interval=int(1000 / base_fps), repeat=False
    )
    ani.save(savepath, writer=animation.PillowWriter(fps=base_fps))

    plt.close(fig)

# combination-based selection helper functions
@dataclass(frozen=True)
class ZScaler:
    mean_: float
    std_: float
    eps: float = 1e-12

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        std = self.std_ if abs(self.std_) > self.eps else 1.0
        return (x - self.mean_) / std
       
def _score_solution(sol, distance_matrix, mcc_per_tree, w_dist, w_perf, dist_scaler, perf_scaler,
                    all_oob_preds, oob_indices_list, y_train, n_classes, mcc_computation):
    sol = list(sol)
    avg_dist = _avg_pairwise_distance(sol, distance_matrix)

    if mcc_computation == "subforest":
        perf = _subforest_oob_mcc(sol, all_oob_preds, oob_indices_list, y_train, n_classes)
    else:
        perf = float(np.nanmean(mcc_per_tree[sol])) if len(sol) else np.nan

    z_dist = float(dist_scaler.transform([avg_dist])[0])
    z_perf = float(perf_scaler.transform([perf])[0]) if np.isfinite(perf) else 0.0

    return float((w_dist * z_dist + w_perf * z_perf) / (w_dist + w_perf))

def _crossover(p1, p2, cut):
    size = len(p1)

    child = p1[:cut]

    for gene in p2[cut:]:
        if gene not in child:
            child.append(gene)

        if len(child) == size:
            return child

    # dann linker Teil von p2
    for gene in p2[:cut]:
        if gene not in child:
            child.append(gene)

        if len(child) == size:
            return child

    return child

def _mutate(child, mutation_probability, n_trees, rng):
    used = set(child)
    for i in range(len(child)):
        if rng.rand() < mutation_probability:
            available = list(set(range(n_trees)) - used)
            if not available:
                continue
            new_gene = int(rng.choice(available))
            used.remove(child[i])
            child[i] = new_gene
            used.add(new_gene)
    return child

def _sus_select(scores, n_select, rng):
    scores = np.asarray(scores, dtype=float)
    if np.all(~np.isfinite(scores)):
        return rng.choice(len(scores), size=n_select, replace=True).tolist()
    min_score = np.nanmin(scores)
    shift = -min_score if min_score < 0 else 0.0
    adj = scores + shift + 1e-12
    total = float(np.sum(adj))
    if total <= 0 or not np.isfinite(total):
        return rng.choice(len(scores), size=n_select, replace=True).tolist()
    step = total / n_select
    start = rng.uniform(0, step)
    points = start + step * np.arange(n_select)

    cum = np.cumsum(adj)
    selected = []
    i = 0
    for p in points:
        while cum[i] < p:
            i += 1
        selected.append(i)
    return selected

def _fit_zscaler(values, eps = 1e-12):
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return ZScaler(mean_=0.0, std_=1.0, eps=eps)
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=0))
    if not np.isfinite(std) or abs(std) <= eps:
        std = 1.0
    return ZScaler(mean_=mean, std_=std, eps=eps)

def _avg_pairwise_distance(sol, distance_matrix):
    sol = list(sol)
    if len(sol) <= 1:
        return 0.0
    sub = distance_matrix[np.ix_(sol, sol)]
    tri = sub[np.triu_indices_from(sub, k=1)]
    if tri.size == 0:
        return 0.0
    return float(np.nanmean(tri))

def _sample_random_solutions(n_trees, subforest_size, n_samples, rng):
    if subforest_size <= 0 or subforest_size > n_trees:
        raise ValueError(f"subforest_size must be in the range [1, {n_trees}]")
    n_samples = int(max(1, n_samples))
    return [
        rng.choice(n_trees, size=subforest_size, replace=False).astype(int).tolist()
        for _ in range(n_samples)
    ]

def _fit_combination_zscalers_from_random_solutions(distance_matrix, subforest_size, mcc_per_tree, all_oob_preds, oob_indices_list, y_train, 
                                                    n_classes, mcc_computation, seed, n_random_solutions=_ZSCALE_N_RANDOM_SOLUTIONS):
    rng = np.random.RandomState(seed + 1337)
    n_trees = int(distance_matrix.shape[0])

    sols = _sample_random_solutions(n_trees, subforest_size, n_random_solutions, rng)

    dist_samples = np.array([_avg_pairwise_distance(sol, distance_matrix) for sol in sols], dtype=float)
    dist_scaler = _fit_zscaler(dist_samples)

    if mcc_computation == "subforest":
        perf_samples = np.array([_subforest_oob_mcc(sol, all_oob_preds, oob_indices_list, y_train, n_classes) for sol in sols])
        perf_scaler = _fit_zscaler(perf_samples)
    else:
        perf_scaler = _fit_zscaler(mcc_per_tree)

    return dist_scaler, perf_scaler

# universal helper functions
def _subforest_oob_mcc(selected, all_oob_preds, oob_indices_list, y_train, n_classes):
    n = y_train.shape[0]
    votes = np.zeros((n, n_classes), dtype=int)
    counts = np.zeros(n, dtype=int)

    for tidx in selected:
        preds = all_oob_preds[tidx]
        oob_idx = oob_indices_list[tidx]
        
        if preds is None:
            continue
            
        votes[oob_idx, preds] += 1
        counts[oob_idx] += 1

    mask = counts > 0
    if np.sum(mask) < 2:
        return np.nan
        
    y_hat = np.argmax(votes[mask], axis=1)
    return float(matthews_corrcoef(y_train[mask], y_hat))

def _precompute_all_oob_predictions(trees, oob_indices_list, X_train):
    all_oob_preds = []
    for tidx, tree in enumerate(trees):
        oob_idx = oob_indices_list[tidx]
        if oob_idx is None or len(oob_idx) == 0:
            all_oob_preds.append(None)
        else:
            # Hier passiert die "teure" Arbeit nur ein einziges Mal pro Baum
            preds = tree.predict(X_train[oob_idx]).astype(int)
            all_oob_preds.append(preds)
    return all_oob_preds

def _validate_subforest_size(subforest_size, n_trees):
    if subforest_size <= 0 or subforest_size > n_trees:
        raise ValueError(f"subforest_size must be in the range [1, {n_trees}]")

def _compute_silhouette_curve(distance_matrix, method, seed):
    n_trees = distance_matrix.shape[0]
    ks = np.arange(1, n_trees + 1, dtype=int)
    scores = np.full(n_trees, np.nan, dtype=float)

    for k in ks:
        try:
            if method == "k-medoid":
                model = KMedoids(
                    n_clusters=int(k),
                    metric="precomputed",
                    random_state=seed
                )
                labels = model.fit_predict(distance_matrix)
            elif method == "agglomerative":
                model = AgglomerativeClustering(
                    n_clusters=int(k),
                    metric="precomputed",
                    linkage="average"
                )
                labels = model.fit_predict(distance_matrix)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # silhouette is only defined for 2..n_trees-1 clusters
            n_labels = len(np.unique(labels))
            if 2 <= n_labels <= (n_trees - 1):
                scores[k - 1] = float(
                    silhouette_score(distance_matrix, labels, metric="precomputed")
                )
        except Exception:
            scores[k - 1] = np.nan

    return ks, scores

def _save_silhouette_curve_plot(ks, scores, method_name, save_dir, name=None, size=None):
    from datetime import datetime
    import os
    
    os.makedirs(save_dir, exist_ok=True)

    run_tag = []
    if name is not None:
        run_tag.append(str(name))
    if size is not None:
        run_tag.append(str(size))
    suffix = ("_" + "_".join(run_tag)) if run_tag else ""

    timestamp = datetime.now().strftime("%M%S")
    savepath = os.path.join(save_dir, f"silhouette_curve_{method_name}{suffix}_{timestamp}.png")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(ks, scores, marker="o", linewidth=2.0, markersize=4.5, color="#1f77b4")
    ax.set_title(f"Silhouette Curve ({method_name})", fontsize=13, pad=10)
    ax.set_xlabel("Number of clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette score", fontsize=11)
    ax.set_xlim(int(np.min(ks)), int(np.max(ks)))
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size > 0:
        y_min = max(-1.0, float(np.min(finite_scores)) - 0.05)
        y_max = min(1.0, float(np.max(finite_scores)) + 0.05)
        ax.set_ylim(y_min, y_max)
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
   
