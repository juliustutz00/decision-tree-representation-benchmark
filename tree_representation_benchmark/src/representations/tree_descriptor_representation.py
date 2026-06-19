import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine

from .base import BaseRepresentation

class TreeDescriptorRepresentation(BaseRepresentation):

    def __init__(self, weights=None, metric="cosine", X=None):
        self.weights = weights if weights is not None else {
            'structure': 0.5,
            'features': 0.3,
            'thresholds': 0.2
        }
        self.metric = metric
        self.X = X
        
        if X is not None:
            C = np.abs(np.corrcoef(X, rowvar=False))
            np.fill_diagonal(C, 1.0)
            self.corr_matrix = np.nan_to_num(C)
        else:
            self.corr_matrix = None

    def represent(self, tree, X_train=None):
        G = self.tree_to_networkx(tree)
        n_nodes = tree.tree_.node_count
        max_depth = tree.tree_.max_depth
        
        leaves = [n for n in G.nodes if G.out_degree(n) == 0]
        root = [n for n in G.nodes if G.in_degree(n) == 0][0]
        
        leaf_depths = []
        leaf_samples = []
        for leaf in leaves:
            path = nx.shortest_path(G, root, leaf)
            depth = len(path) - 1
            leaf_depths.append(depth)
            leaf_samples.append(tree.tree_.n_node_samples[leaf])
            
        avg_path_len = np.mean(leaf_depths) if leaf_depths else 0
        total_samples = tree.tree_.n_node_samples[root]
        weighted_path_len = np.sum(np.array(leaf_depths) * np.array(leaf_samples)) / total_samples
        balance_std = np.std(leaf_depths) if leaf_depths else 0
        
        depths_dict = nx.single_source_shortest_path_length(G, root)
        depth_counts = np.bincount(list(depths_dict.values()))
        max_width = np.max(depth_counts) if len(depth_counts) > 0 else 0
        leaf_impurities = tree.tree_.impurity[leaves]
        avg_leaf_impurity = np.mean(leaf_impurities)

        n_features = X_train.shape[1]
        used_features = tree.tree_.feature
        internal_mask = used_features >= 0
        
        unique_features_used = len(np.unique(used_features[internal_mask]))
        feature_diversity = unique_features_used / n_features if n_features > 0 else 0
        
        # structure metrics
        structural_metrics = np.array([
            np.log1p(n_nodes),
            np.log1p(max_depth),
            np.log1p(max_width),
            np.log1p(avg_path_len),
            np.log1p(weighted_path_len),
            balance_std,
            avg_leaf_impurity,
            feature_diversity
        ])

        # feature metrics
        feat_importance = np.nan_to_num(tree.feature_importances_)
        
        feature_counts = np.zeros(n_features)
        if np.any(internal_mask):
            counts = np.bincount(used_features[internal_mask], minlength=n_features)
            feature_counts = counts.astype(float) / ((n_nodes - 1) / 2)

        # thresholds
        thresholds = np.full(n_features, np.nan)
        if np.any(internal_mask):
            for f in range(n_features):
                f_mask = (used_features == f)
                if np.any(f_mask):
                    f_thresholds = tree.tree_.threshold[f_mask]
                    mean_threshold = np.mean(f_thresholds)
                    min_val, max_val = X_train[:, f].min(), X_train[:, f].max()
                    thresholds[f] = (mean_threshold - min_val) / (max_val - min_val) if max_val > min_val else 0.5

        return {
            "structure": structural_metrics,
            "importance": feat_importance,
            "counts": feature_counts,
            "thresholds": thresholds
        }

    def similarity(self, rep_a, rep_b):
        sim_struct = 1 - cosine(rep_a["structure"], rep_b["structure"])
        

        sim_imp = self._soft_cosine_similarity(rep_a["importance"], rep_b["importance"])
        sim_cnt = self._soft_cosine_similarity(rep_a["counts"], rep_b["counts"])
        sim_feat = (sim_imp + sim_cnt) / 2

        mask_a, mask_b = ~np.isnan(rep_a["thresholds"]), ~np.isnan(rep_b["thresholds"])
        intersection = mask_a & mask_b
        if np.any(intersection):
            diffs = np.abs(rep_a["thresholds"][intersection] - rep_b["thresholds"][intersection])
            sim_thresh = 1.0 - np.mean(diffs)
        else:
            sim_thresh = 0.0

        return (self.weights['structure'] * sim_struct +
                self.weights['features'] * sim_feat +
                self.weights['thresholds'] * sim_thresh)
    

    def _soft_cosine_similarity(self, v1, v2):
        if self.corr_matrix is None:
            return 1 - cosine(v1, v2) if np.any(v1) and np.any(v2) else 0.0
        
        # (v1^T * C * v2) / sqrt((v1^T * C * v1) * (v2^T * C * v2))
        v1_C = v1 @ self.corr_matrix
        v2_C = v2 @ self.corr_matrix
        
        numerator = v1_C @ v2
        denominator = np.sqrt((v1_C @ v1) * (v2_C @ v2))
        
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
    def tree_to_networkx(self, tree):
        G = nx.DiGraph()
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        G.add_nodes_from(range(n_nodes))

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:
                G.add_edge(i, children_left[i])
                G.add_edge(i, children_right[i])

        return G
        
