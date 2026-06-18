import os
import networkx as nx
import numpy as np
import pandas as pd
import re
import time
from collections import defaultdict
from hashlib import md5


FOLDER = r"x" #replace with actual path


PAIRS = [
    ("graph1smalllicit.dot",      "graph2smallilicit.dot",   "Small"),
    ("graph1medium1licit.dot", "graph2medium1ilicit.dot", "Medium"),
    ("graph1biglicit.dot",       "graph2bigilicit.dot",         "Large"),
]

OUTPUT_FILE = os.path.join(FOLDER, "x.txt") #replace with actual output file name
BTC_THRESHOLDS = [0.00000001, 0.0001, 0.001, 0.01]

SPLIT_COLORS = { #colors can be changed
    "SR_R0":   "lightgreen",
    "SR_R25":  "lightblue",
    "SR_R50":  "gold",
    "SR_R75":  "orange",
    "SR_R100": "pink",
    "SR_none": "gray85"
}

WL_BORDER_STYLES = ["solid", "dashed", "bold", "dotted"]

def parse_dot_manually(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    values, labels = {}, {}
    label_pattern = re.compile(r'"([^"]+)"\s*\[label="([^"]+)"\]')
    for node_id, label in label_pattern.findall(content):
        labels[node_id] = label
        val_match = re.search(r"Value:\s*([\d\.eE\+\-]+)\s*BTC", label)
        if val_match:
            try:
                values[node_id] = float(val_match.group(1))
            except:
                values[node_id] = None
    edge_pattern = re.compile(r'"([^"]+)"\s*(?:->|--)\s*"([^"]+)"')
    edges = edge_pattern.findall(content)
    return edges, values, labels

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

def _wl_relabel_dag(G, initial_labels, n_iter=3):
    node_list = list(nx.topological_sort(G))
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    current_labels = {n: initial_labels[n] for n in G.nodes()}
    label_sequences = {node_to_idx[n]: [current_labels[n]] for n in G.nodes()}
    for _ in range(n_iter):
        new_labels = {}
        for node in node_list:
            pred_labels = sorted(current_labels[p] for p in G.predecessors(node))
            succ_labels = sorted(current_labels[s] for s in G.successors(node))
            agg = f"{current_labels[node]}|P:{','.join(pred_labels)}|C:{','.join(succ_labels)}"
            new_labels[node] = md5(agg.encode()).hexdigest()[:8]
        current_labels = new_labels
        for node in G.nodes():
            label_sequences[node_to_idx[node]].append(current_labels[node])
    return label_sequences, idx_to_node

def _wl_feature_vector(label_sequences):
    counter = defaultdict(int)
    for seq in label_sequences.values():
        for iteration, label in enumerate(seq):
            counter[f"i{iteration}_{label}"] += 1
    return counter

def _cosine_similarity(v1, v2):
    keys = set(v1) | set(v2)
    a = np.array([v1.get(k, 0) for k in keys], dtype=float)
    b = np.array([v2.get(k, 0) for k in keys], dtype=float)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0


def build_nx_graph(edge_data):
    if not edge_data:
        return None
    G = nx.DiGraph()
    G.add_edges_from(edge_data)
    if not nx.is_directed_acyclic_graph(G):
        while True:
            try:
                cycle = nx.find_cycle(G)
                G.remove_edge(cycle[-1][0], cycle[-1][1])
            except nx.NetworkXNoCycle:
                break
    return G

def main():
    WL_ITER = 3
    start_total = time.time()
    report_lines = [
        f"{'='*65}",
        f"Split Ratio WL Similarity  F(v) = #small_outputs / outdegree(v)",
        f"WL Iterations: {WL_ITER}",
        f"{'='*65}",
    ]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for file1, file2, size_label in PAIRS:
            edges1, values1, labels1 = parse_dot_manually(os.path.join(FOLDER, file1))
            edges2, values2, labels2 = parse_dot_manually(os.path.join(FOLDER, file2))
            G_meta1 = build_nx_graph(edges1)
            G_meta2 = build_nx_graph(edges2)
            if G_meta1 is None or G_meta2 is None:
                report_lines.append(f"\n[{size_label}] Error: graph construction failed.\n")
                continue
            report_lines += [
                f"\n{'#'*65}",
                f"  GRAPH SIZE: {size_label}  |  {file1} vs {file2}",
                f"  Nodes G1: {G_meta1.number_of_nodes()}  Edges G1: {G_meta1.number_of_edges()}",
                f"  Nodes G2: {G_meta2.number_of_nodes()}  Edges G2: {G_meta2.number_of_edges()}",
                f"{'#'*65}",
                f"  {'Threshold (BTC)':<20} {'WL Similarity':>15}  {'Time (s)':>10}",
                f"  {'-'*50}",
            ]
            del G_meta1, G_meta2
            for btc_thr in BTC_THRESHOLDS:
                t0 = time.perf_counter()
                G1 = build_nx_graph(edges1)
                G2 = build_nx_graph(edges2)
                init1 = {n: _split_ratio_label(n, G1, values1, btc_thr) for n in G1.nodes()}
                init2 = {n: _split_ratio_label(n, G2, values2, btc_thr) for n in G2.nodes()}
                seq1, idx_to_node1 = _wl_relabel_dag(G1, init1, n_iter=WL_ITER)
                seq2, idx_to_node2 = _wl_relabel_dag(G2, init2, n_iter=WL_ITER)
                all_final_labels = sorted(
                    {s[-1] for s in seq1.values()} |
                    {s[-1] for s in seq2.values()}
                )
                wl_style_map = {
                    lbl: WL_BORDER_STYLES[i % len(WL_BORDER_STYLES)]
                    for i, lbl in enumerate(all_final_labels)
                }
                if size_label == "Small":
                    for idx, seq in seq1.items():
                        node = idx_to_node1[idx]
                        G1.nodes[node].update({
                            "label":     f'"{labels1.get(node, node)}"',
                            "fillcolor": SPLIT_COLORS.get(init1[node], "gray85"),
                            "style":     "filled," + wl_style_map[seq[-1]],
                            "penwidth":  "2", "shape": "box",
                            "fontsize":  "10", "color": "black", "fontcolor": "black"
                        })
                    for idx, seq in seq2.items():
                        node = idx_to_node2[idx]
                        G2.nodes[node].update({
                            "label":     f'"{labels2.get(node, node)}"',
                            "fillcolor": SPLIT_COLORS.get(init2[node], "gray85"),
                            "style":     "filled," + wl_style_map[seq[-1]],
                            "penwidth":  "2", "shape": "box",
                            "fontsize":  "10", "color": "black", "fontcolor": "black"
                        })
                    for G in [G1, G2]:
                        G.graph["pad"] = "0.4"
                        G.graph["margin"] = "0.2"
                    thr_str = str(btc_thr).replace('.', '_')
                    nx.drawing.nx_pydot.write_dot(
                        G1, os.path.join(FOLDER, f"graph1_wl_split_{size_label}_thr_{thr_str}.dot"))
                    nx.drawing.nx_pydot.write_dot(
                        G2, os.path.join(FOLDER, f"graph2_wl_split_{size_label}_thr_{thr_str}.dot"))
                fv1, fv2 = _wl_feature_vector(seq1), _wl_feature_vector(seq2)
                sim     = _cosine_similarity(fv1, fv2)
                elapsed = time.perf_counter() - t0
                report_lines.append(
                    f"  <= {btc_thr:<17} {sim:>15.4f}  {elapsed:>10.5f}"
                )
                print(f"  {size_label} <= {btc_thr}  sim={sim:.4f}  time={elapsed:.5f}s")
            report_lines.append(f"  {'='*50}")
        total_elapsed = time.time() - start_total
        report_lines += [
            "",
            f"{'='*65}",
            f"Total Execution Time: {total_elapsed:.4f} s",
            f"{'='*65}",
        ]
        report = "\n".join(report_lines) + "\n"
        print("\n" + report)
        f.write(report)
if __name__ == "__main__":
    main()