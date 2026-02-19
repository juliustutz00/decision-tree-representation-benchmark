from zss import Node, simple_distance


def compute_structural_difference(original, variation, X_train):
    """
    Compute tree edit distance between two sklearn trees via zss.

    Node substitution costs:
    - exact match: 0
    - insertion/deletion or leaf/internal mismatch: 1
    - different split features: 2
    - same feature, different thresholds: normalized threshold gap (capped at 0.5)
    """

    def build_zss_tree(tree, node_id=0):
        if node_id == -1:
            return None

        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        if feature == -2:
            label = "Leaf"
        else:
            label = f"f{feature}:{threshold:.3f}"

        zss_node = Node(label)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        for child_id in [left_child, right_child]:
            if child_id != -1:
                zss_node.addkid(build_zss_tree(tree, child_id))

        return zss_node

    def substitution_cost(a, b):
        label_a, label_b = a, b
        if label_a == '' or label_b == '': # insertion / deletion
            return 1
        elif (label_a == label_b): # equal nodes
            return 0
        elif "Leaf" in (label_a, label_b): # internal and leaf node
            return 1

        fa, ta = label_a.split(":")
        fb, tb = label_b.split(":")
        fa, fb = int(fa[1:]), int(fb[1:])
        ta, tb = float(ta), float(tb)

        if fa != fb: # 2 internal nodes, different feature
            return 2

        feature_values = X_train[:, fa]
        f_range = feature_values.max() - feature_values.min()
        diff = abs(ta - tb) / f_range if f_range > 0 else 0.0
        return min(diff, 0.5) # 2 internal nodes, same feature, threshold difference
    
    original_zss = build_zss_tree(original.tree_)
    variation_zss = build_zss_tree(variation.tree_)
    distance = simple_distance(original_zss, variation_zss, label_dist=substitution_cost)
    return distance
