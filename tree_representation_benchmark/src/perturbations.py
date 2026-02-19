import copy
import numpy as np

"""
This file is a collection of perturbations that can be applied to a DecisionTreeClassifier from scikit-learn using the __setstate__ function

The functions modify structural/split properties and then refresh node statistics
through `update_subtree(...)`.
"""

def change_threshold(tree, template_tree, X_train, y_train, features_info, intensity, strength, seed=None):
    """
    Perturb thresholds of a fraction (`intensity`; [0, n_nodes]) of internal nodes.

    Continuous/integer features get additive noise; categorical/binary features
    get reassigned to a sampled category value.
    """
    rng = np.random.RandomState(seed)
    tree_copy = copy.deepcopy(tree)
    tree_state = tree_copy.tree_.__getstate__()
    internal_nodes = [i for i, n in enumerate(tree_state["nodes"]) if n["feature"] >= 0]
    n_change = int(len(internal_nodes) * intensity) 
    change_nodes = rng.choice(internal_nodes, n_change, replace=False)

    for node_idx in change_nodes:
        feature = tree_state["nodes"][node_idx]["feature"]
        feature_column = X_train[:, feature]
        feature_type =  str(features_info.loc[feature, "type"]).lower()
        
        if feature_type in ["real", "continuous", "integer"]:
            rng_value = np.max(feature_column) - np.min(feature_column)
            if rng_value == 0:
                continue
            perturbation = rng.normal(scale=rng_value)
            # this way a random value is added or subtracted from the threshold of the node
            # by not randomly changing the threshold to any value in [np.min(feature_column), np.max(feature_column)] it's possible to use the strength parameter in future work
            tree_state["nodes"][node_idx]["threshold"] += perturbation
            if feature_type == "integer":
                tree_state["nodes"][node_idx]["threshold"] = int(round(tree_state["nodes"][node_idx]["threshold"]))
        elif feature_type in ["categorical", "binary"]:
            unique_values = np.unique(feature_column)
            if len(unique_values) <= 1:
                continue
            new_threshold = rng.choice(unique_values)
            tree_state["nodes"][node_idx]["threshold"] = new_threshold
        else:
            raise ValueError("Unsupported feature type for perturbation.")


    update_subtree(0, X_train, y_train, features_info, np.arange(len(X_train)), tree_state)
    tree_copy.tree_.__setstate__(tree_state)
    return tree_copy


def change_feature(tree, template_tree, X_train, y_train, features_info, intensity, strength, seed=None):
    """
    Replace split features on [0, n_nodes] selected internal nodes and recompute thresholds
    using local Gini-based search.
    """
    rng = np.random.RandomState(seed)
    tree_copy = copy.deepcopy(tree)
    tree_state = tree_copy.tree_.__getstate__()
    n_features = X_train.shape[1]
    internal_nodes = [i for i, node in enumerate(tree_state["nodes"]) if node["feature"] >= 0]
    n_change = int(len(internal_nodes) * intensity)
    if n_change == 0:
        return tree_copy
    change_nodes = rng.choice(internal_nodes, n_change, replace=False)

    for node_idx in change_nodes:
        random_feature = rng.randint(0, n_features)
        random_feature_type =  str(features_info.loc[random_feature, "type"]).lower()
        sample_indices = np.arange(len(X_train)) 
        sample_indices_at_node = __get_sample_indices_at_node(0, node_idx, sample_indices, tree_state, X_train, features_info).astype(int)
        optimal_threshold = __find_optimal_threshold(X_train, y_train, random_feature, random_feature_type, sample_indices_at_node)
        tree_state["nodes"][node_idx]["feature"] = random_feature
        tree_state["nodes"][node_idx]["threshold"] = optimal_threshold

    update_subtree(0, X_train, y_train, features_info, np.arange(len(X_train)), tree_state)
    tree_copy.tree_.__setstate__(tree_state)
    return tree_copy

def swap_nodes(tree, template_tree, X_train, y_train, features_info, intensity, strength, seed=None):
    """Swap feature/threshold definitions between [0, n_nodes/2] random internal-node pairs."""
    rng = np.random.RandomState(seed)
    tree_copy = copy.deepcopy(tree)
    tree_state = tree_copy.tree_.__getstate__()
    internal_nodes = [i for i, node in enumerate(tree_state["nodes"]) if node["feature"] >= 0]
    n_swaps = int((len(internal_nodes) * intensity) / 2) 
    if n_swaps == 0:
        return tree_copy
    swap_pairs = rng.choice(internal_nodes, size=2 * n_swaps, replace=False)

    for i in range(n_swaps):
        node_a = swap_pairs[2 * i]
        node_b = swap_pairs[2 * i + 1]
        tmp_feature = tree_state["nodes"][node_a]["feature"]
        tmp_threshold = tree_state["nodes"][node_a]["threshold"]
        tree_state["nodes"][node_a]["feature"] = tree_state["nodes"][node_b]["feature"]
        tree_state["nodes"][node_a]["threshold"] = tree_state["nodes"][node_b][
            "threshold"
        ]
        tree_state["nodes"][node_b]["feature"] = tmp_feature
        tree_state["nodes"][node_b]["threshold"] = tmp_threshold

    update_subtree(0, X_train, y_train, features_info, np.arange(len(X_train)), tree_state)
    tree_copy.tree_.__setstate__(tree_state)
    return tree_copy


def remove_nodes(tree, template_tree, X_train, y_train, features_info, intensity, strength, seed=None):
    """Iteratively prune [0, n_nodes/2] pairs of leaf children until target removal intensity is met."""
    rng = np.random.RandomState(seed)
    tree_copy = copy.deepcopy(tree)
    tree_state = tree_copy.tree_.__getstate__()
    removal_count = 0
    n_removals = int((tree_state["node_count"] - 1) * intensity)
    if n_removals == 0:
        return tree_copy

    while removal_count < n_removals:
        # parent_nodes refers to parent nodes with 2 children that can be removed (as they are both leafs) such that the parent node will become a leaf
        parent_nodes = __find_parent_with_two_leafs(tree_state)
        chosen_parent_node = rng.choice(parent_nodes)
        __remove_children(chosen_parent_node, tree_state)
        removal_count += 2

    update_subtree(0, X_train, y_train, features_info, np.arange(len(X_train)), tree_state)
    tree_copy.tree_.__setstate__(tree_state)
    return tree_copy


# since this implementation might cause memory issues due to its internal cython representation, the following requirement has to be met: n_nodes(template_tree) > (n_nodes(tree)-1) * 2 
def add_nodes(tree, template_tree, X_train, y_train, features_info, intensity, strength, seed=None):
    """
    Expand [0, n_nodes/2] selected leaves by adding two children each.

    Note: since this implementation might cause memory issues due to its internal cython representation, 
    the following requirement has to be met: n_nodes(template_tree) > (n_nodes(tree)-1) * 2 
    """
    rng = np.random.RandomState(seed)
    tree_copy = copy.deepcopy(tree)
    template_tree_copy = copy.deepcopy(template_tree)
    tree_state = tree_copy.tree_.__getstate__()
    template_tree_state = template_tree_copy.tree_.__getstate__()
    if not len(template_tree_state['nodes']) > ((len(tree_state['nodes']) - 1) * 2):
        raise ValueError("add_node: Since this implementation might cause memory issues due to its internal cython representation the following requirement has to be met: n_nodes(template_tree) > (n_nodes(tree)-1) * 2")
    addition_count = 0
    n_additions = int((tree_state["node_count"] - 1) * intensity)
    if n_additions == 0:
        return tree_copy

    while addition_count < n_additions:
        leaf_nodes = __find_leaf_nodes(tree_state)
        chosen_leaf = rng.choice(leaf_nodes)
        __add_children(chosen_leaf, tree_state, X_train, y_train, features_info, rng, strength)
        addition_count += 2

    update_subtree(0, X_train, y_train, features_info, np.arange(len(X_train)), tree_state)
    template_tree_copy.tree_.__setstate__(tree_state)
    return template_tree_copy


def combined_perturbations(tree, template_tree, X_train, y_train, features_info, intensity, strength, seed=None):
    """Apply a random subset of [1, 6] perturbations in sequence."""
    rng = np.random.RandomState(seed)
    tree_copy = copy.deepcopy(tree)
    perturbations = [change_threshold, change_feature, swap_nodes, remove_nodes, add_nodes]
    num_perturbations = rng.randint(1, 6)
    available_perturbations = perturbations.copy()
    selected_perturbations = []
    for _ in range(num_perturbations):
        chosen_perturbation = rng.choice(available_perturbations)
        selected_perturbations.append(chosen_perturbation)
        if chosen_perturbation == add_nodes:
            available_perturbations.remove(add_nodes)
    for perturbation in selected_perturbations:
        tree_copy = perturbation(tree_copy, template_tree, X_train, y_train, features_info, intensity, strength, seed)
    return tree_copy


def update_subtree(node_id, X, y, features_info, sample_indices, tree_state, parent_class_distribution=None):
    """
    Recompute node statistics (class distribution, sample counts, impurity)
    recursively after structure/split edits.
    """
    nodes = tree_state["nodes"]
    node = nodes[node_id]
    feature = node["feature"]
    threshold = node["threshold"]

    # if the node is empty (n_node_samples == 0) the prediction value of its parent node is being used; if the root is empty a prediciton value np.zeroes(size=n_classes) is used
    unique_classes = np.unique(y)
    class_counts = np.array([np.sum(y[sample_indices] == c) for c in unique_classes])
    total = class_counts.sum()
    if total > 0:
        class_distribution = class_counts / total
    else:
        class_distribution = (
            parent_class_distribution.copy()
            if parent_class_distribution is not None
            else np.zeros_like(class_counts)
        )

    tree_state["values"][node_id][0] = class_distribution
    tree_state["nodes"][node_id]["n_node_samples"] = len(sample_indices)
    tree_state["nodes"][node_id]["weighted_n_node_samples"] = float(len(sample_indices))
    tree_state["nodes"][node_id]["impurity"] = __compute_gini(class_counts)

    if feature >= 0:
        X_node = X[sample_indices, feature]
        feature_type =  str(features_info.loc[feature, "type"]).lower()
        if feature_type in ["real", "continuous", "integer"]:
            left_indices = sample_indices[X_node <= threshold]
            right_indices = sample_indices[X_node > threshold]
        elif feature_type in ["categorical", "binary"]:
            left_indices = sample_indices[X_node == threshold]
            right_indices = sample_indices[X_node != threshold]
        else:
            raise ValueError("Unsupported feature type for perturbation.")
        
        update_subtree(
            node["left_child"], X, y, features_info, left_indices, tree_state, class_distribution
        )
        update_subtree(
            node["right_child"], X, y, features_info, right_indices, tree_state, class_distribution
        )


def __compute_gini(class_counts):
    total = sum(class_counts)
    if total == 0:
        return 0.0
    probs = [count / total for count in class_counts]
    return 1.0 - sum(p**2 for p in probs)

def __find_optimal_threshold(X, y, feature_idx, feature_type, sample_indices):
    if len(sample_indices) == 0:
        return np.mean(X[:, feature_idx])
    feature_values = X[sample_indices, feature_idx]

    unique_values = np.unique(feature_values)
    if len(unique_values) == 1:
        return unique_values[0]
    best_gini = float('inf')
    best_threshold = None

    if feature_type in ["real", "continuous", "integer"]:
        for i in range(1, len(unique_values)):
            threshold = (unique_values[i-1] + unique_values[i]) / 2
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            left_indices = sample_indices[left_mask]
            right_indices = sample_indices[right_mask]
            
            left_gini = __compute_gini(np.bincount(y[left_indices]))
            right_gini = __compute_gini(np.bincount(y[right_indices]))
            total_samples = len(sample_indices)
            weighted_gini = (len(left_indices) / total_samples) * left_gini + (len(right_indices) / total_samples) * right_gini
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_threshold = threshold
        
        best_threshold = int(round(best_threshold)) if feature_type == "integer" else best_threshold

    elif feature_type in ["categorical", "binary"]:
        for value in unique_values:
            left_mask = feature_values == value
            right_mask = ~left_mask
            left_indices = sample_indices[left_mask]
            right_indices = sample_indices[right_mask]
            
            left_gini = __compute_gini(np.bincount(y[left_indices]))
            right_gini = __compute_gini(np.bincount(y[right_indices]))
            total_samples = len(sample_indices)
            weighted_gini = (len(left_indices) / total_samples) * left_gini + (len(right_indices) / total_samples) * right_gini
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_threshold = value

    else:
        raise ValueError("Unsupported feature type for perturbation.")
    
    return best_threshold

def __get_sample_indices_at_node(current_node_idx, searched_node_idx, sample_indices, tree_state, X_train, features_info):
    if current_node_idx == searched_node_idx:
        if sample_indices is None or len(sample_indices) == 0:
            sample_indices = np.array([])
        return sample_indices
    node = tree_state["nodes"][current_node_idx]
    feature = node["feature"]
    threshold = node["threshold"]

    if feature >= 0:
        X_node = X_train[sample_indices, feature]
        feature_type =  str(features_info.loc[feature, "type"]).lower()
        # if the node is internal, split the samples based on the threshold
        if feature_type in ["real", "continuous", "integer"]:
            left_mask = X_node <= threshold
            right_mask = ~left_mask
        elif feature_type in ["categorical", "binary"]:
            left_mask = X_node == threshold
            right_mask = ~left_mask
        else:
            raise ValueError("Unsupported feature type for perturbation.")

        # get the sample indices for left and right children
        left_indices = sample_indices[left_mask]
        right_indices = sample_indices[right_mask]

        # recursively traverse to left and right child nodes
        if node["left_child"] >= 0:
            left_indices = __get_sample_indices_at_node(node["left_child"], searched_node_idx, left_indices, tree_state, X_train, features_info)
        if node["right_child"] >= 0:
            right_indices = __get_sample_indices_at_node(node["right_child"], searched_node_idx, right_indices, tree_state, X_train, features_info)
        return np.concatenate((left_indices, right_indices))
    else:
        return np.array([])

def __find_parent_with_two_leafs(tree_state):
    parents_with_two_leafs = []
    for node_idx, node in enumerate(tree_state["nodes"]):
        left_node = node["left_child"]
        right_node = node["right_child"]
        if left_node == -1 or right_node == -1:
            continue
        if (
            tree_state["nodes"][left_node]["left_child"] == -1
            and tree_state["nodes"][left_node]["right_child"] == -1
            and tree_state["nodes"][right_node]["left_child"] == -1
            and tree_state["nodes"][right_node]["right_child"] == -1
        ):
            parents_with_two_leafs.append(node_idx)
    return parents_with_two_leafs


def __find_leaf_nodes(tree_state):
    leaf_nodes = []
    for node_idx, node in enumerate(tree_state["nodes"]):
        if node["left_child"] == -1 and node["right_child"] == -1:
            leaf_nodes.append(node_idx)
    return leaf_nodes


def __remove_children(parent_node, tree_state):
    left_child_idx = tree_state["nodes"][parent_node][0]
    right_child_idx = tree_state["nodes"][parent_node][1]
    to_delete = sorted([left_child_idx, right_child_idx], reverse=True)
    tree_state["nodes"] = np.delete(tree_state["nodes"], to_delete, axis=0)
    tree_state["values"] = np.delete(tree_state["values"], to_delete, axis=0)

    for node in tree_state["nodes"]:
        for child in ["left_child", "right_child"]:
            if node[child] in to_delete:
                node[child] = -1
            else:
                for deleted_idx in to_delete:
                    if node[child] > deleted_idx:
                        node[child] -= 1

    new_parent = parent_node
    for d in to_delete:
        if parent_node > d:
            new_parent -= 1

    tree_state["nodes"][new_parent]["left_child"] = -1
    tree_state["nodes"][new_parent]["right_child"] = -1
    tree_state["nodes"][new_parent]["feature"] = -2
    tree_state["nodes"][new_parent]["threshold"] = -2.0

    tree_state["node_count"] += -2
    tree_state["max_depth"] = __calculate_depth(0, tree_state)


def __add_children(leaf_node, tree_state, X_train, y_train, features_info, rng, strength):
    new_left_node_idx = len(tree_state["nodes"])
    new_right_node_idx = new_left_node_idx + 1
    node_dtype = tree_state["nodes"].dtype
    new_nodes = np.array(
        [(-1, -1, -2, -2.0, 0, 0, 0.0, 0), (-1, -1, -2, -2.0, 0, 0, 0.0, 0)],
        dtype=node_dtype,
    )
    # it's not possible to use np.concatenate/append as the dtype behaves weird
    combined_nodes = np.empty(
        tree_state["nodes"].shape[0] + new_nodes.shape[0], dtype=node_dtype
    )
    combined_nodes[: tree_state["nodes"].shape[0]] = tree_state["nodes"]
    combined_nodes[tree_state["nodes"].shape[0] :] = new_nodes
    tree_state["nodes"] = combined_nodes
    leaf_node_value = tree_state["values"][leaf_node].copy()
    tree_state["values"] = np.append(
        tree_state["values"], [leaf_node_value, leaf_node_value], axis=0
    )
    tree_state["nodes"][leaf_node]["left_child"] = new_left_node_idx
    tree_state["nodes"][leaf_node]["right_child"] = new_right_node_idx
    n_features = X_train.shape[1]
    random_feature_new_parent = rng.randint(0, n_features)
    random_feature_new_parent_type =  str(features_info.loc[random_feature_new_parent, "type"]).lower()
    sample_indices = np.arange(len(X_train)) 
    sample_indices_at_node = __get_sample_indices_at_node(0, leaf_node, sample_indices, tree_state, X_train, features_info).astype(int)
    optimal_threshold_new_parent = __find_optimal_threshold(X_train, y_train, random_feature_new_parent, random_feature_new_parent_type, sample_indices_at_node)
    tree_state["nodes"][leaf_node]["feature"] = random_feature_new_parent
    tree_state["nodes"][leaf_node]["threshold"] = optimal_threshold_new_parent

    tree_state["node_count"] += 2
    tree_state["max_depth"] = __calculate_depth(0, tree_state)


def __calculate_depth(idx, tree_state):
    # depth is calculated according to sklearn's implementation
    if (
        tree_state["nodes"][idx]["left_child"] == -1
        and tree_state["nodes"][idx]["right_child"] == -1
    ):
        return 1
    return 1 + max(
        __calculate_depth(tree_state["nodes"][idx]["left_child"], tree_state),
        __calculate_depth(tree_state["nodes"][idx]["right_child"], tree_state),
    )
