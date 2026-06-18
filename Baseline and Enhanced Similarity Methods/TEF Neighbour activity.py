import os
import re
import time
from collections import defaultdict, Counter
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cosine

FOLDER = r"x" #replace with actual path

PAIRS = [
    ("graph1smalllicit.dot",      "graph2smallilicit.dot",   "Small"),
    ("graph1medium1licit.dot", "graph2medium1ilicit.dot", "Medium"),
    ("graph1biglicit.dot",       "graph2bigilicit.dot",         "Large"),
]

OUTPUT_FILE = os.path.join(FOLDER, "x.txt") #replace with actual output file name

TEMPORAL_BINS = [ #colors can be changed
    (1,            "ultra_dense"),
    (6,            "dense"),
    (24,           "moderate"),
    (float("inf"), "sparse"),
]

NEIGHBORHOOD_PALETTES = {
    "ultra_dense": ["#dcedc8", "#c5e1a5", "#aed581"],
    "dense":       ["#fff9c4", "#fff59d", "#fff176"],
    "moderate":    ["#bbdefb", "#90caf9", "#64b5f6"],
    "sparse":      ["#cfd8dc", "#b0bec5", "#90a4ae"],
    "none":        ["#f5f5f5", "#eeeeee", "#e0e0e0"],
}

def _temporal_label(avg_delta_hours):
    for threshold, label in TEMPORAL_BINS:
        if avg_delta_hours <= threshold:
            return label
    return "sparse"

def parse_dot_manually(filepath):
    if not os.path.exists(filepath):
        return [], {}, None
    G_pydot = nx.drawing.nx_pydot.read_dot(filepath)
    timestamps = {}
    for node in G_pydot.nodes():
        lbl = G_pydot.nodes[node].get("label", "")
        lbl = " ".join(lbl) if isinstance(lbl, list) else lbl
        m = re.search(r'Time:\s+([\d\-: ]+)', str(lbl))
        if m:
            try:
                timestamps[node] = pd.to_datetime(m.group(1).strip())
            except Exception:
                timestamps[node] = None
        else:
            timestamps[node] = None
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    edge_pattern = re.compile(r'"([^"]+)"\s*(?:->|--)\s*"([^"]+)"')
    edges = edge_pattern.findall(content)
    return edges, timestamps, G_pydot

def build_nx_graph(edge_data, base_pydot_G):
    if not edge_data:
        return None
    G = nx.DiGraph()
    G.add_edges_from(edge_data)
    for node in G.nodes():
        if node in base_pydot_G.nodes():
            G.nodes[node].update(base_pydot_G.nodes[node])
    return G
def _get_node_neighborhood_signature(node, G, timestamps):
    current_time = timestamps.get(node)
    if current_time is None:
        return "T_none"
    diffs = []
    for pred in G.predecessors(node):
        p_t = timestamps.get(pred)
        if p_t:
            d = (current_time - p_t).total_seconds() / 3600.0
            if d >= 0:
                diffs.append(d)
    for succ in G.successors(node):
        s_t = timestamps.get(succ)
        if s_t:
            d = (s_t - current_time).total_seconds() / 3600.0
            if d >= 0:
                diffs.append(d)
    if not diffs:
        return "T_none"
    counts = defaultdict(int)
    for d in diffs:
        counts[_temporal_label(d)] += 1
    bin_names = [label for _, label in TEMPORAL_BINS]
    return "T_" + "_".join(f"{n[0].upper()}{counts.get(n, 0)}" for n in bin_names)

def get_topology_metrics(G, timestamps):
    if G is None or len(G) == 0:
        return None
    deg_hist = Counter([d for _, d in G.degree()])
    G_undirected = G.to_undirected()
    avg_clust = nx.average_clustering(G_undirected)
    centrality = sorted(list(nx.degree_centrality(G).values()))
    try:
        path_len = nx.average_shortest_path_length(G_undirected) if nx.is_connected(G_undirected) else 0
    except Exception:
        path_len = 0
    signatures = {n: _get_node_neighborhood_signature(n, G, timestamps) for n in G.nodes()}
    global_signature_distribution = Counter(signatures.values())
    bin_counts = {"ultra_dense": 0, "dense": 0, "moderate": 0, "sparse": 0}
    node_primary_labels = {}
    for node in G.nodes():
        current_time = timestamps.get(node)
        if current_time is None:
            node_primary_labels[node] = "none"
            continue
        causal_diffs = []
        for pred in G.predecessors(node):
            pred_time = timestamps.get(pred)
            if pred_time is not None:
                delta = (current_time - pred_time).total_seconds() / 3600.0
                if delta >= 0:
                    causal_diffs.append(delta)
        for succ in G.successors(node):
            succ_time = timestamps.get(succ)
            if succ_time is not None:
                delta = (succ_time - current_time).total_seconds() / 3600.0
                if delta >= 0:
                    causal_diffs.append(delta)
        if not causal_diffs:
            node_primary_labels[node] = "none"
            continue
        avg_delta = np.mean(causal_diffs)
        label = _temporal_label(avg_delta)
        node_primary_labels[node] = label
        bin_counts[label] += 1
    return {
        "node_count": len(G.nodes()), "edge_count": len(G.edges()),
        "deg_hist": deg_hist, "clustering": avg_clust,
        "centrality": centrality, "path_len": path_len,
        "global_signatures": global_signature_distribution,
        "node_primary_labels": node_primary_labels,
        **bin_counts,
    }

def compare_metrics(m1, m2):
    all_degs = set(m1["deg_hist"].keys()) | set(m2["deg_hist"].keys())
    v1 = np.array([m1["deg_hist"].get(d, 0) for d in all_degs], dtype=float)
    v2 = np.array([m2["deg_hist"].get(d, 0) for d in all_degs], dtype=float)
    deg_score   = 1.0 - min(1.0, euclidean(v1 / (sum(v1) or 1), v2 / (sum(v2) or 1)))
    clust_score = 1.0 - abs(m1["clustering"] - m2["clustering"])
    min_l = min(len(m1["centrality"]), len(m2["centrality"]))
    c_sim = 1.0 - cosine(m1["centrality"][:min_l], m2["centrality"][:min_l]) if min_l > 0 else 0.0
    max_path  = max(m1["path_len"], m2["path_len"], 1)
    path_score = 1.0 - (abs(m1["path_len"] - m2["path_len"]) / max_path)
    all_sigs = set(m1["global_signatures"].keys()) | set(m2["global_signatures"].keys())
    sv1 = np.array([m1["global_signatures"].get(s, 0) for s in all_sigs], dtype=float)
    sv2 = np.array([m2["global_signatures"].get(s, 0) for s in all_sigs], dtype=float)
    sig_score = 1.0 - min(1.0, euclidean(sv1 / (sum(sv1) or 1), sv2 / (sum(sv2) or 1)))
    return {
        "degree_score": deg_score, "clustering_score": clust_score,
        "centrality_score": c_sim, "path_score": path_score,
        "signature_topology_score": sig_score,
        "composite_match": (deg_score + clust_score + c_sim + path_score + sig_score) / 5.0,
    }

def apply_hybrid_neighborhood_coloring(G, metrics):
    lbl_map  = metrics["node_primary_labels"]
    cent_dict = nx.degree_centrality(G)
    scores   = list(cent_dict.values())
    q25, q75 = np.percentile(scores, [25, 75]) if len(scores) > 1 else [0.0, 0.0]
    for node in G.nodes():
        primary_label = lbl_map.get(node, "none")
        node_importance = cent_dict.get(node, 0.0)
        shade_idx = 0 if node_importance <= q25 else (1 if node_importance < q75 else 2)
        fill_color = NEIGHBORHOOD_PALETTES[primary_label][shade_idx]
        deg = G.degree(node)
        border_width = "1.0" if deg <= 2 else ("2.5" if deg <= 5 else "4.0")
        G.nodes[node].update({
            "style": "filled", "fillcolor": fill_color,
            "color": "#000000", "fontcolor": "#000000",
            "shape": "box", "penwidth": border_width
        })
    return G

def main():
    report_lines = [
        "=" * 65,
        "NEIGHBOURHOOD TOPOLOGY SIMILARITY — ALL GRAPH SIZES",
        "=" * 65,
    ]
    for file1, file2, size_label in PAIRS:
        print(f"Running {size_label}...")
        t0 = time.perf_counter()

        edges1, times1, raw_G1 = parse_dot_manually(os.path.join(FOLDER, file1))
        edges2, times2, raw_G2 = parse_dot_manually(os.path.join(FOLDER, file2))
        G1 = build_nx_graph(edges1, raw_G1)
        G2 = build_nx_graph(edges2, raw_G2)
        m1 = get_topology_metrics(G1, times1)
        m2 = get_topology_metrics(G2, times2)
        elapsed = time.perf_counter() - t0

        if not m1 or not m2:
            report_lines.append(f"\n[{size_label}] Error: insufficient graph data.\n")
            continue

        scores = compare_metrics(m1, m2)
        report_lines += [
            f"\n{'#'*65}",
            f"  GRAPH SIZE: {size_label}  |  {file1} vs {file2}",
            f"{'#'*65}",
            f"  GRAPH 1 — Nodes: {m1['node_count']}  Edges: {m1['edge_count']}",
            f"    Ultra-Dense: {m1['ultra_dense']}  Dense: {m1['dense']}  Moderate: {m1['moderate']}  Sparse: {m1['sparse']}",
            f"  GRAPH 2 — Nodes: {m2['node_count']}  Edges: {m2['edge_count']}",
            f"    Ultra-Dense: {m2['ultra_dense']}  Dense: {m2['dense']}  Moderate: {m2['moderate']}  Sparse: {m2['sparse']}",
            f"  {'-'*55}",
            f"  Degree Alignment:      {scores['degree_score']:.4f}",
            f"  Clustering Alignment:  {scores['clustering_score']:.4f}",
            f"  Centrality Alignment:  {scores['centrality_score']:.4f}",
            f"  Path Length Alignment: {scores['path_score']:.4f}",
            f"  Neighbourhood Role:    {scores['signature_topology_score']:.4f}",
            f"  {'-'*55}",
            f"  >>> Overall Similarity: {scores['composite_match']:.4f}",
            f"  Time: {elapsed:.5f} s",
        ]
        print(f"  {size_label}: sim={scores['composite_match']:.4f}  time={elapsed:.5f}s")
        if size_label == "Small":
            nx.drawing.nx_pydot.write_dot(
                apply_hybrid_neighborhood_coloring(G1.copy(), m1),
                os.path.join(FOLDER, "graph1_neighborhood_colored.dot"))
            nx.drawing.nx_pydot.write_dot(
                apply_hybrid_neighborhood_coloring(G2.copy(), m2),
                os.path.join(FOLDER, "graph2_neighborhood_colored.dot"))

    report_lines += ["", "=" * 65]
    report = "\n".join(report_lines) + "\n"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()