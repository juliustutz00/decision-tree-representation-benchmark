import numpy as np
from zss import Node, simple_distance
import ot


def compute_structural_difference(original, variation, X_train):
    
    def build_zss_tree(tree, node_id=0):
        """Baue rekursiv einen zss.Node-Baum aus einem sklearn tree_ Objekt."""
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
            return 1#1.5
        elif (label_a == label_b): # equal nodes
            return 0
        elif "Leaf" in (label_a, label_b): # internal and leaf node
            return 1#2.0

        fa, ta = label_a.split(":")
        fb, tb = label_b.split(":")
        fa, fb = int(fa[1:]), int(fb[1:])
        ta, tb = float(ta), float(tb)

        if fa != fb: # 2 internal nodes, different feature
            return 2#1.0

        feature_values = X_train[:, fa]
        f_range = feature_values.max() - feature_values.min()
        diff = abs(ta - tb) / f_range if f_range > 0 else 0.0
        return min(diff, 0.5) # 2 internal nodes, same feature, threshold difference
    
    original_zss = build_zss_tree(original.tree_)
    variation_zss = build_zss_tree(variation.tree_)
    distance = simple_distance(original_zss, variation_zss, label_dist=substitution_cost)
    return distance


def compute_feature_importance_difference(original, variation, X_train, correlation_adjustment=False):

    def feature_importance_vector(tree, n_features: int) -> np.ndarray:
        """Return feature_importances_ padded/truncated to n_features, normalized to sum=1 if possible."""
        if tree is None or not hasattr(tree, "feature_importances_"):
            return np.zeros(n_features, dtype=float)

        fi = np.asarray(tree.feature_importances_, dtype=float).ravel()
        if fi.size < n_features:
            fi = np.pad(fi, (0, n_features - fi.size), mode="constant")
        else:
            fi = fi[:n_features]

        fi = np.nan_to_num(fi, nan=0.0, posinf=0.0, neginf=0.0)
        fi[fi < 0] = 0.0

        s = float(fi.sum())
        if s > 0:
            fi = fi / s
        return fi

    n_features = X_train.shape[1]
    p = feature_importance_vector(original, n_features)
    q = feature_importance_vector(variation, n_features)

    if not correlation_adjustment:
        return np.linalg.norm(p - q, ord=1)
    else:
        corr = np.corrcoef(X_train, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        D = 1 - np.abs(corr)

        p = np.maximum(p, 0)
        q = np.maximum(q, 0)
        if p.sum() > 0:
            p = p / p.sum()
        else: 
            p = np.ones_like(p) / len(p)

        if q.sum() > 0:
            q = q / q.sum()
        else:
            q = np.ones_like(q) / len(q)

        return ot.emd2(p, q, D)
