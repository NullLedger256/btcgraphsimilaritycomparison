import os
import networkx as nx
from collections import Counter
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import time

FOLDER = r"x" #replace with actual path

# Defined the three pairs for processing
PAIRS = [
    ("graph1smalllicit.dot",      "graph2smallilicit.dot",   "Small"),
    ("graph1medium1licit.dot", "graph2medium1ilicit.dot", "Medium"),
    ("graph1biglicit.dot",       "graph2bigilicit.dot",         "Large"),
]

OUTPUT_FILE = os.path.join(FOLDER, "x.txt") #replace with actual output file name

def normalized_hist(counter, max_val=10):
    vec = [counter.get(i, 0) for i in range(max_val + 1)]
    total = sum(vec)
    return np.array(vec) / total if total > 0 else np.zeros(max_val + 1)

def centrality_vector(G, centrality_func):
    c = centrality_func(G)
    return np.array(sorted(c.values())) if c else np.array([])

def path_vec(G):
    if not nx.is_connected(G) and len(G) > 0:
        G = G.subgraph(max(nx.connected_components(G), key=len))
    if len(G) == 0:
        return np.zeros(11)
    spl = dict(nx.shortest_path_length(G))
    lengths = [l for s in spl.values() for l in s.values() if 0 < l <= 10]
    return normalized_hist(Counter(lengths), max_val=10)

def similarity_metrics(G1, G2):
    deg1, deg2 = Counter([d for _, d in G1.degree()]), Counter([d for _, d in G2.degree()])
    v1, v2 = normalized_hist(deg1), normalized_hist(deg2)
    degree_dist = euclidean(v1, v2)
    c1 = nx.average_clustering(G1) if len(G1) > 0 else 0
    c2 = nx.average_clustering(G2) if len(G2) > 0 else 0
    clustering_diff = abs(c1 - c2)
    cv1, cv2 = centrality_vector(G1, nx.degree_centrality), centrality_vector(G2, nx.degree_centrality)
    min_len = min(len(cv1), len(cv2))
    centrality_sim = 1 - cosine(cv1[:min_len], cv2[:min_len]) if min_len > 0 else 0
    p1, p2 = path_vec(G1), path_vec(G2)
    path_dist = euclidean(p1, p2)
    composite = degree_dist + clustering_diff + (1 - centrality_sim) + path_dist
    return degree_dist, clustering_diff, centrality_sim, path_dist, composite

def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"{'Size':<10} {'DegreeDist':>12} {'ClustDiff':>12} {'CentSim':>12} {'PathDist':>12} {'Composite':>12}\n")
        f.write("-" * 75 + "\n")

        for file1, file2, size_label in PAIRS:
            try:
                G1 = nx.Graph(nx.drawing.nx_pydot.read_dot(os.path.join(FOLDER, file1)))
                G2 = nx.Graph(nx.drawing.nx_pydot.read_dot(os.path.join(FOLDER, file2)))
                results = similarity_metrics(G1, G2)
                f.write(f"{size_label:<10} {results[0]:>12.4f} {results[1]:>12.4f} {results[2]:>12.4f} {results[3]:>12.4f} {results[4]:>12.4f}\n")
                print(f"Processed {size_label}...")
            except Exception as e:
                print(f"Error on {size_label}: {e}")

if __name__ == "__main__":
    main()