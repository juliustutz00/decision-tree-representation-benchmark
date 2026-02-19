import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine, euclidean

from .base import BaseRepresentation

# novel representation
class TreeDescriptorRepresentation(BaseRepresentation):
    """Feature + structure metric vector representation for decision trees."""

    def __init__(self, weights=None, metric="cosine"):
        self.weights = weights
        self.metric = metric

    def represent(self, tree, X_train=None):
        """
        Encode tree into a fixed-length numeric descriptor vector.

        Source: novel representation
        """
        G = self.tree_to_networkx(tree)
        n_nodes = tree.tree_.node_count
        max_depth = tree.tree_.max_depth
        
        leaves = [n for n in G.nodes if G.out_degree(n) == 0]
        root = [n for n in G.nodes if G.in_degree(n) == 0][0]
        
        leaf_depths = []
        leaf_samples = []
        for leaf in leaves:
            depth = len(nx.shortest_path(G, root, leaf)) - 1
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

        n_features = tree.tree_.n_features
        used_features = tree.tree_.feature
        internal_mask = used_features >= 0
        
        unique_features_used = len(np.unique(used_features[internal_mask]))
        feature_diversity = unique_features_used / n_features if n_features > 0 else 0

        feat_importance = np.nan_to_num(tree.feature_importances_)

        structural_metrics = np.array([
            np.log1p(n_nodes),
            np.log1p(max_depth),
            np.log1p(max_width),
            np.log1p(avg_path_len),
            np.log1p(weighted_path_len),
            balance_std,
            avg_leaf_impurity
        ])

        feature_counts = np.zeros(n_features, dtype=float)
        if np.any(internal_mask):
            counts = np.bincount(used_features[internal_mask], minlength=n_features)
            feature_counts = counts.astype(float) / ((n_nodes-1) / 2)

        thresholds = np.zeros(n_features, dtype=float)
        if np.any(internal_mask):
            for f in range(n_features):
                f_thresholds = tree.tree_.threshold[used_features == f]
                if len(f_thresholds) > 0:
                    thresholds[f] = np.mean(f_thresholds)

        combined_vector = np.concatenate([
            structural_metrics, 
            feat_importance,
            feature_counts,
            np.array([feature_diversity]), 
            thresholds
        ])

        if self.weights is None:
            self.weights = np.ones(len(combined_vector))
            
        return combined_vector * self.weights
    
    def represent_TF(self, tree, X_train=None):
        # graph metrics are chosen accordingly to the Toplogical Forest Paper + feature stuff
        # however, some of the metrics do not make sense for for trees, therefore this function is not used
        G = self.tree_to_networkx(tree)

        # graph-information
        if not nx.is_strongly_connected(G):
            G_sub = max(nx.strongly_connected_components(G), key=len)
            G_sub = G.subgraph(G_sub).copy()
            diameter = nx.diameter(G_sub)
        else:
            diameter = nx.diameter(G)
        number_of_nodes = G.number_of_nodes()
        number_of_edges = G.number_of_edges()
        triad_keys = ['003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300']
        three_node_motifs = [nx.triadic_census(G)[k] for k in triad_keys]
        clustering_coefficient = nx.average_clustering(G)

        # node-information
        avg_in_degree = np.mean(list(dict(G.in_degree()).values()))
        avg_out_degree = np.mean(list(dict(G.out_degree()).values()))
        avg_degree = np.mean(list(dict(G.degree()).values()))
        if nx.is_strongly_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.strongly_connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subG)
        avg_betweenness_centrality = np.mean(list(nx.betweenness_centrality(G, normalized=True).values()))

        metrics = np.array([
            diameter,
            number_of_nodes,
            number_of_edges,
            *three_node_motifs,
            clustering_coefficient,
            avg_in_degree,
            avg_out_degree,
            avg_degree,
            avg_path_length,
            avg_betweenness_centrality
        ], dtype=float)

        # feature-information
        n_features = tree.tree_.n_features
        used_features = tree.tree_.feature
        internal_mask = used_features >= 0

        feature_counts = np.zeros(n_features, dtype=float)
        if np.any(internal_mask):
            counts = np.bincount(used_features[internal_mask], minlength=n_features)
            feature_counts[:len(counts)] = counts

        # Feature importances
        feat_importance = tree.feature_importances_
        feat_importance = np.nan_to_num(feat_importance)

        # threshold-statistics
        thresholds = np.zeros(n_features, dtype=float)
        if np.any(internal_mask):
            for f in range(n_features):
                f_thresholds = tree.tree_.threshold[used_features == f]
                if len(f_thresholds) > 0:
                    thresholds[f] = np.mean(f_thresholds) 

        combined_vector = np.concatenate([
            np.array(metrics, dtype=float),
            feature_counts,
            feat_importance,
            thresholds
        ])

        if self.weights is None:
            base_weights = np.array([0.1, 0.1, 0.1, *([0.1 / 16] * 16), 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            self.weights = np.concatenate([
                base_weights,
                (np.ones(n_features * 3) / 3)  # Feature counts, importance, thresholds
            ])
        combined_vector *= self.weights
        return combined_vector
        
    def similarity(self, representation_a, representation_b):
        """Similarity via cosine or inverse-Euclidean transform."""
        if self.metric == "cosine":
            return 1 - cosine(representation_a, representation_b)
        elif self.metric == "euclidean":
            return 1 / (1 + euclidean(representation_a, representation_b))
        else:
            raise ValueError("Undefined metric.")


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
