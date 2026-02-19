import numpy as np
import networkx as nx
import kmapper as km
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from openTSNE import TSNE

from .base import BaseRepresentation

class TopologicalForestRepresentation(BaseRepresentation):
    """
    Topological-graph-inspired representation that first calculates a feature graph and then computes metrics on it.

    Source: https://github.com/cakcora/MultiverseJ/blob/master/python/MultiverseBinaryCode.py
    """

    def __init__(self, tree_vectors):
        if tree_vectors is None:
            return
        self.tree_vectors = tree_vectors
        random_state = 1618033
        scaler = MinMaxScaler(feature_range=(0, 1))
        tree_vectors_scaled = scaler.fit_transform(self.tree_vectors)
        mapper = km.KeplerMapper()
        projected = TSNE(n_components=2, random_state=random_state, n_jobs=-1, learning_rate="auto").fit(np.array(tree_vectors_scaled))
        self.graph = mapper.map(projected, tree_vectors_scaled, cover=km.Cover(n_cubes=10, perc_overlap=0.6), clusterer=KMeans(n_clusters=5, random_state=random_state))
        self.G = km.adapter.to_networkx(self.graph)

    def represent(self, tree, X_train): 
        """Returns feature vector per tree, calculated on a simple feature graph."""
        G = nx.DiGraph()
        tree_ = tree.tree_
        features = tree_.feature
        triad_keys = ['003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300']

        # Recursive traversal
        def add_edges(node_idx):
            feature_idx = tree_.feature[node_idx]
            if feature_idx == -2:
                return 

            left_child = tree_.children_left[node_idx]
            right_child = tree_.children_right[node_idx]

            for child in [left_child, right_child]:
                if child != -1 and features[child] != -2:
                    child_feature_idx = features[child]
                    G.add_edge(feature_idx, child_feature_idx)
                    add_edges(child)

        # start from root
        add_edges(0)  
        
        G.remove_nodes_from([n for n in G.nodes if n < 0])

        if len(G.nodes) == 0:
            diameter = 0
            number_of_nodes = 0
            number_of_edges = 0
            three_node_motifs = [0] * len(triad_keys)
            clustering_coefficient = 0
            avg_in_degree = 0
            avg_out_degree = 0
            avg_degree = 0
            avg_path_length = 0
            avg_betweenness_centrality = 0
        else:
            # graph-information
            if not nx.is_strongly_connected(G):
                G_sub = max(nx.strongly_connected_components(G), key=len)
                G_sub = G.subgraph(G_sub).copy()
                diameter = nx.diameter(G_sub)
            else:
                diameter = nx.diameter(G)
            number_of_nodes = G.number_of_nodes()
            number_of_edges = G.number_of_edges()
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

        feature_vector = np.array([
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

        return feature_vector
    
    def similarity(self, representation_a, representation_b):
        """Compares tree indices through Mapper-cluster shortest-path distance."""
        
        # for testing purposes the initial design of the abstract representation class is abused in the sense that 
        # all tree representations first have to be computed before similarities can be calculated
        # representation_a and _b have to be indices in the tree_vectors list
        if self.tree_vectors == []:
            raise ValueError("To use this representation, all vector representations of the trees first have to be computed.")
        clusters_a = [n for n, members in self.graph["nodes"].items() if representation_a in members]
        clusters_b = [n for n, members in self.graph["nodes"].items() if representation_b in members]

        if set(clusters_a) & set(clusters_b):
            return 1.0

        # shortest path between any of the clusters in graph distance
        min_dist = np.inf
        for ca in clusters_a:
            for cb in clusters_b:
                try:
                    d = nx.shortest_path_length(self.G, ca, cb)
                    min_dist = min(min_dist, d)
                except nx.NetworkXNoPath:
                    continue

        if np.isinf(min_dist):
            return 0.0
        else:
            return 1 / (1 + min_dist)
