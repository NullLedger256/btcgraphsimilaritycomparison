import os
import re
import time
import networkx as nx
import numpy as np
from collections import Counter
from scipy.spatial.distance import euclidean, cosine


FOLDER = r"x" #replace with actual path

PAIRS = [
    ("graph1smalllicit.dot",      "graph2smallilicit.dot",   "Small"),
    ("graph1medium1licit.dot", "graph2medium1ilicit.dot", "Medium"),
    ("graph1biglicit.dot",       "graph2bigilicit.dot",         "Large"),
]

BTC_THRESHOLDS = [0.00000001, 0.0001, 0.001, 0.01]
OUTPUT_FILE = os.path.join(FOLDER, "x") #replace with actual output file name


SR_PALETTES = { #colors can be changed
    "SR_R0":   ["#dcedc8", "#c5e1a5", "#aed581"],
    "SR_R25":  ["#fff9c4", "#fff59d", "#fff176"],
    "SR_R50":  ["#bbdefb", "#90caf9", "#64b5f6"],
    "SR_R75":  ["#e1bee7", "#ce93d8", "#ba68c8"],
    "SR_R100": ["#f8bbd0", "#f48fb1", "#f06292"],
    "SR_none": ["#f5f5f5", "#eeeeee", "#e0e0e0"]
}


def parse_dot_manually(filepath):
    if not os.path.exists(filepath):
        return [], {}, {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    values, labels = {}, {}
    label_pattern = re.compile(r'"([^"]+)"\s*\[label="([^"]+)"\]')
    for node_id, label in label_pattern.findall(content):
        labels[node_id] = label
        val_match = re.search(r"Value:\s*([\d\.eE\+\-]+)\s*BTC", label)
        values[node_id] = float(val_match.group(1)) if val_match else None
    edges = re.compile(r'"([^"]+)"\s*(?:->|--)\s*"([^"]+)"').findall(content)
    return edges, values, labels

def build_nx_graph(edge_data, labels):
    if not edge_data:
        return None
    G = nx.DiGraph()
    G.add_edges_from(edge_data)
    for node in G.nodes():
        G.nodes[node]['label'] = labels.get(node, str(node))
    while True:
        try:
            cycle = nx.find_cycle(G)
            G.remove_edge(cycle[-1][0], cycle[-1][1])
        except nx.NetworkXNoCycle:
            break
    return G

def _split_ratio_label(node, G, values, btc_threshold):
    successors = list(G.successors(node))
    if not successors:
        return "SR_none"
    out_vals = [values.get(s) for s in successors if values.get(s) is not None]
    if not out_vals:
        return "SR_none"
    ratio = sum(1 for v in out_vals if v <= btc_threshold) / len(successors)
    if ratio == 0.0:    return "SR_R0"
    elif ratio <= 0.25: return "SR_R25"
    elif ratio <= 0.5:  return "SR_R50"
    elif ratio <= 0.75: return "SR_R75"
    else:               return "SR_R100"

def apply_split_ratio_coloring(G, values, btc_threshold):
    cent_dict = nx.degree_centrality(G)
    scores = list(cent_dict.values())
    q25, q75 = np.percentile(scores, [25, 75]) if len(scores) > 1 else [0.0, 0.0]
    for node in G.nodes():
        sr_label = _split_ratio_label(node, G, values, btc_threshold)
        node_importance = cent_dict.get(node, 0.0)
        shade_idx = 0 if node_importance <= q25 else (1 if node_importance < q75 else 2)
        deg = G.degree(node)
        G.nodes[node].update({
            "style": "filled",
            "fillcolor": SR_PALETTES.get(sr_label, SR_PALETTES["SR_none"])[shade_idx],
            "color": "#000000", "fontcolor": "#000000", "shape": "box",
            "penwidth": "4.0" if deg > 10 else ("2.5" if deg > 5 else "1.2")
        })
    return G

def get_topology_metrics(G, values, btc_threshold):
    G_undirected = G.to_undirected()
    return {
        "deg_hist":   Counter([d for _, d in G.degree()]),
        "clustering": nx.average_clustering(G_undirected),
        "centrality": sorted(list(nx.degree_centrality(G).values())),
        "path_len":   nx.average_shortest_path_length(G_undirected)
                      if nx.is_connected(G_undirected) else 0,
        "split_dist": Counter(
            _split_ratio_label(n, G, values, btc_threshold) for n in G.nodes()),
    }

def compare_metrics(m1, m2):
    all_degs = set(m1["deg_hist"].keys()) | set(m2["deg_hist"].keys())
    v1 = np.array([m1["deg_hist"].get(d, 0) for d in all_degs], dtype=float)
    v2 = np.array([m2["deg_hist"].get(d, 0) for d in all_degs], dtype=float)
    deg_score   = 1.0 - min(1.0, euclidean(v1 / (sum(v1) or 1), v2 / (sum(v2) or 1)))
    clust_score = 1.0 - abs(m1["clustering"] - m2["clustering"])
    min_l = min(len(m1["centrality"]), len(m2["centrality"]))
    c_sim = 1.0 - cosine(m1["centrality"][:min_l], m2["centrality"][:min_l]) if min_l > 0 else 0.0
    max_path    = max(m1["path_len"], m2["path_len"], 1)
    path_score  = 1.0 - (abs(m1["path_len"] - m2["path_len"]) / max_path)
    all_sigs = set(m1["split_dist"].keys()) | set(m2["split_dist"].keys())
    sv1 = np.array([m1["split_dist"].get(s, 0) for s in all_sigs], dtype=float)
    sv2 = np.array([m2["split_dist"].get(s, 0) for s in all_sigs], dtype=float)
    split_score = 1.0 - min(1.0, euclidean(sv1 / (sum(sv1) or 1), sv2 / (sum(sv2) or 1)))
    return {
        "degree_score":     deg_score,
        "clustering_score": clust_score,
        "centrality_score": c_sim,
        "path_score":       path_score,
        "split_ratio_score":split_score,
        "composite_match":  (deg_score + clust_score + c_sim + path_score + split_score) / 5.0,
    }

def main():
    lines = ["FRAGMENTATION SPLIT RATIO — TOPOLOGICAL SIMILARITY REPORT", "=" * 65]

    for file1, file2, size_label in PAIRS:
        edges1, vals1, labs1 = parse_dot_manually(os.path.join(FOLDER, file1))
        edges2, vals2, labs2 = parse_dot_manually(os.path.join(FOLDER, file2))
        G1 = build_nx_graph(edges1, labs1)
        G2 = build_nx_graph(edges2, labs2)

        if G1 is None or G2 is None:
            lines.append(f"\n[{size_label}] Error: graph construction failed.\n")
            continue
        lines += [
            f"\n{'#'*60}",
            f"  GRAPH SIZE: {size_label}  |  {file1} vs {file2}",
            f"  Nodes G1: {G1.number_of_nodes()}  Edges G1: {G1.number_of_edges()}",
            f"  Nodes G2: {G2.number_of_nodes()}  Edges G2: {G2.number_of_edges()}",
            f"{'#'*60}",
        ]
        for btc_thr in BTC_THRESHOLDS:
            t0 = time.perf_counter()
            m1 = get_topology_metrics(G1, vals1, btc_thr)
            m2 = get_topology_metrics(G2, vals2, btc_thr)
            s  = compare_metrics(m1, m2)
            elapsed = time.perf_counter() - t0

            lines += [
                f"\n  Threshold <= {btc_thr:.8f} BTC",
                f"  Degree Alignment:      {s['degree_score']:.4f}",
                f"  Clustering Alignment:  {s['clustering_score']:.4f}",
                f"  Centrality Alignment:  {s['centrality_score']:.4f}",
                f"  Path Length Alignment: {s['path_score']:.4f}",
                f"  Split Ratio Alignment: {s['split_ratio_score']:.4f}",
                f"  ---------------------------------------------------",
                f"  >>> Overall Similarity: {s['composite_match']:.4f}",
                f"  Time: {elapsed:.5f} s",
            ]
            if size_label == "Small":
                g1_vis = apply_split_ratio_coloring(G1.copy(), vals1, btc_thr)
                g2_vis = apply_split_ratio_coloring(G2.copy(), vals2, btc_thr)
                nx.drawing.nx_pydot.write_dot(
                    g1_vis, os.path.join(FOLDER, f"g1_sr_{btc_thr}.dot"))
                nx.drawing.nx_pydot.write_dot(
                    g2_vis, os.path.join(FOLDER, f"g2_sr_{btc_thr}.dot"))

        lines.append("\n" + "=" * 65)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



if __name__ == "__main__":
    main()