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

TEMPORAL_BINS = [
    (1,            "ultra_dense"),
    (6,            "dense"),
    (24,           "moderate"),
    (float("inf"), "sparse"),
]

HEURISTIC_COLORS = {   #colors can be changed
    "ultra_dense": "gold",
    "dense":       "orange",
    "moderate":    "lightblue",
    "sparse":      "lightgreen",
    "none":        "gray85"
}

WL_BORDER_STYLES = ["solid", "dashed", "bold", "dotted"]

def _temporal_label(avg_delta_hours):
    for threshold, label in TEMPORAL_BINS:
        if avg_delta_hours <= threshold:
            return label
    return "sparse"

def parse_dot_manually(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    timestamps, labels = {}, {}
    label_pattern = re.compile(r'"([^"]+)"\s*\[label="([^"]+)"\]')
    for node_id, label in label_pattern.findall(content):
        labels[node_id] = label
        time_match = re.search(r"Time:\s+([\d\-: ]+)", label)
        if time_match:
            try:
                timestamps[node_id] = pd.to_datetime(time_match.group(1).strip())
            except:
                timestamps[node_id] = None
    edge_pattern = re.compile(r'"([^"]+)"\s*(?:->|--)\s*"([^"]+)"')
    edges = edge_pattern.findall(content)
    return edges, timestamps, labels


def _initial_node_label(node, G, timestamps):
    current_time = timestamps.get(node)
    direct_neighbours = list(G.predecessors(node)) + list(G.successors(node))
    if current_time is None or not direct_neighbours:
        return "T_none"
    diffs = []
    for nb in direct_neighbours:
        nb_time = timestamps.get(nb)
        if nb_time is not None:
            diffs.append(abs((nb_time - current_time).total_seconds()) / 3600.0)
    if not diffs:
        return "T_none"
    counts = defaultdict(int)
    for d in diffs:
        counts[_temporal_label(d)] += 1
    bin_names = [label for _, label in TEMPORAL_BINS]
    return "T_" + "_".join(f"{name[0].upper()}{counts.get(name, 0)}" for name in bin_names)

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

def heuristic_category(init_label):
    if init_label == "T_none":
        return "none"
    counts = {}
    for name, short in zip(["ultra_dense", "dense", "moderate", "sparse"], ["U", "D", "M", "S"]):
        m = re.search(rf"{short}(\d+)", init_label)
        counts[name] = int(m.group(1)) if m else 0
    return max(counts.items(), key=lambda x: x[1])[0]


def bin_counts(init_labels):
    totals = defaultdict(int)
    bin_names = [lbl for _, lbl in TEMPORAL_BINS]
    for lbl in init_labels.values():
        if lbl == "T_none":
            continue
        for name, short in zip(bin_names, ["U", "D", "M", "S"]):
            match = re.search(rf"{short}(\d+)", lbl)
            if match:
                totals[name] += int(match.group(1))
    return [totals.get(name, 0) for name in bin_names]


def main():
    WL_ITER = 3
    results = []

    for file1, file2, size_label in PAIRS:
        edges1, ts1, labels1 = parse_dot_manually(os.path.join(FOLDER, file1))
        edges2, ts2, labels2 = parse_dot_manually(os.path.join(FOLDER, file2))
        G1 = build_nx_graph(edges1)
        G2 = build_nx_graph(edges2)

        if G1 is None or G2 is None:
            continue

        t0 = time.time()
        init1 = {n: _initial_node_label(n, G1, ts1) for n in G1.nodes()}
        init2 = {n: _initial_node_label(n, G2, ts2) for n in G2.nodes()}
        seq1, _ = _wl_relabel_dag(G1, init1, n_iter=WL_ITER)
        seq2, _ = _wl_relabel_dag(G2, init2, n_iter=WL_ITER)

        fv1, fv2 = _wl_feature_vector(seq1), _wl_feature_vector(seq2)
        sim     = _cosine_similarity(fv1, fv2)
        elapsed = time.time() - t0

        results.append({
            "label": size_label,
            "bins1": bin_counts(init1),
            "bins2": bin_counts(init2),
            "sim":   sim,
            "time":  elapsed,
        })

    # === Output Table ===
    print("\n" + "="*85)
    print(f"{'Graph':<8} | {'G':<2} | {'U-Dense':<8} | {'Dense':<6} | {'Mod':<4} | {'Sparse':<6} | {'Similarity':<10} | {'Time (s)':<8}")
    print("-"*85)
    for res in results:
        b1, b2 = res['bins1'], res['bins2']
        print(f"{res['label']:<8} | G1 | {b1[0]:<8} | {b1[1]:<6} | {b1[2]:<4} | {b1[3]:<6} | {res['sim']:.4f}     | {res['time']:.5f}")
        print(f"{'':<8} | G2 | {b2[0]:<8} | {b2[1]:<6} | {b2[2]:<4} | {b2[3]:<6} | {'':<10} | {'':<8}")
        print("-"*85)
if __name__ == "__main__":
    main()