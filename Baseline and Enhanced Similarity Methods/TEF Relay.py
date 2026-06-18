import os
import re
import time
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
from scipy.spatial.distance import euclidean, cosine


FOLDER = r"x" #replace with actual path

PAIRS = [
    ("graph1smalllicit.dot",      "graph2smallilicit.dot",   "Small"),
    ("graph1medium1licit.dot", "graph2medium1ilicit.dot", "Medium"),
    ("graph1biglicit.dot",       "graph2bigilicit.dot",         "Large"),
]

THRESHOLDS = [6, 24, 48, 72, 96]
OUTPUT_FILE = os.path.join(FOLDER, "x") #replace with actual output file name


HYBRID_PALETTES = { #colors can be changed
    "source":   ["#e2f0d9", "#a9d18e", "#385723"],
    "relay":    ["#fff2cc", "#ffd966", "#e6b800"],
    "target":   ["#ddebf7", "#9bc2e6", "#1f4e78"],
    "inactive": ["#f2f2f2", "#d9d9d9", "#595959"]
}



def load_and_parse_dot(filepath):
    if not os.path.exists(filepath):
        return nx.DiGraph(), {}
    G_dir = nx.DiGraph(nx.drawing.nx_pydot.read_dot(filepath))
    timestamps = {}
    for node in G_dir.nodes():
        lbl = G_dir.nodes[node].get("label", "")
        lbl = " ".join(lbl) if isinstance(lbl, list) else lbl
        m = re.search(r'Time:\s+([\d\-: ]+)', str(lbl))
        timestamps[node] = pd.to_datetime(m.group(1)) if m else None
    return G_dir.copy(), timestamps

def get_valid_relay_nodes_and_roles(G, timestamps, max_hours):
    relay_nodes = set()
    active_sources = set()
    active_targets = set()
    valid_edges = []
    for middle_node in G.nodes():
        m_time = timestamps.get(middle_node)
        if m_time is None or pd.isna(m_time):
            continue
        preds = list(G.predecessors(middle_node))
        succs = list(G.successors(middle_node))
        if preds and succs:
            for u in preds:
                u_time = timestamps.get(u)
                if u_time is None or pd.isna(u_time):
                    continue
                if m_time >= u_time:
                    for v in succs:
                        v_time = timestamps.get(v)
                        if v_time is None or pd.isna(v_time):
                            continue
                        if v_time >= m_time:
                            delay_hours = (v_time - m_time).total_seconds() / 3600.0
                            if delay_hours <= max_hours:
                                relay_nodes.add(middle_node)
                                active_sources.add(u)
                                active_targets.add(v)
                                valid_edges.append((u, middle_node))
                                valid_edges.append((middle_node, v))

    valid_edges = list(set(valid_edges))
    node_roles = {}
    for node in G.nodes():
        if node in relay_nodes:
            node_roles[node] = "relay"
        elif node in active_sources:
            node_roles[node] = "source"
        elif node in active_targets:
            node_roles[node] = "target"
        else:
            node_roles[node] = "inactive"

    return valid_edges, node_roles

def normalized_hist(counter, max_val=10):
    vec = [counter.get(i, 0) for i in range(max_val + 1)]
    total = sum(vec)
    return np.array(vec) / total if total > 0 else np.zeros(max_val + 1)
def get_topology_metrics(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    if len(G) == 0:
        return None
    in_v  = normalized_hist(Counter([d for _, d in G.in_degree()]))
    out_v = normalized_hist(Counter([d for _, d in G.out_degree()]))
    clust = nx.average_clustering(G) if len(G) > 0 else 0
    c_dict = nx.in_degree_centrality(G) if len(G) > 0 else {}
    centrality_profile = sorted(list(c_dict.values()))
    if len(G) > 0:
        try:
            spl = dict(nx.all_pairs_shortest_path_length(G, cutoff=10))
            lengths = [l for s in spl.values() for l in s.values() if 0 < l <= 10]
            path_vec = normalized_hist(Counter(lengths), max_val=10)
        except:
            path_vec = np.zeros(11)
    else:
        path_vec = np.zeros(11)
    relay_density = len(edges) / len(G.nodes()) if len(G.nodes()) > 0 else 0
    return {
        "in_v": in_v, "out_v": out_v, "clustering": clust,
        "centrality": centrality_profile, "path_vec": path_vec,
        "relay_density": relay_density,
        "edge_count": len(edges), "node_count": len(G.nodes()),
    }

def compare_metrics(m1, m2):
    degree_mismatch = (euclidean(m1["in_v"], m2["in_v"]) + euclidean(m1["out_v"], m2["out_v"])) / 2
    degree_score    = 1.0 - min(1.0, degree_mismatch)
    clustering_score= 1.0 - min(1.0, abs(m1["clustering"] - m2["clustering"]))
    min_l = min(len(m1["centrality"]), len(m2["centrality"]))
    c_sim = 1.0 - cosine(m1["centrality"][:min_l], m2["centrality"][:min_l]) if min_l > 0 else 0.0
    centrality_score= min(1.0, max(0.0, c_sim))
    path_mismatch   = euclidean(m1["path_vec"], m2["path_vec"])
    path_score      = 1.0 - min(1.0, path_mismatch)
    relay_diff      = abs(m1["relay_density"] - m2["relay_density"])
    relay_score     = 1.0 - min(1.0, relay_diff)
    total_mismatch  = degree_mismatch + abs(m1["clustering"] - m2["clustering"]) + \
                      (1.0 - centrality_score) + path_mismatch + relay_diff
    overall_similarity = 1.0 - (total_mismatch / 5.0)
    return degree_score, clustering_score, centrality_score, path_score, relay_score, overall_similarity

def apply_visualization(base_G, filter_edges, roles):
    vis_G = base_G.copy()
    sub_G = nx.DiGraph()
    sub_G.add_edges_from(filter_edges)
    cent_dict = nx.in_degree_centrality(sub_G) if len(sub_G) > 0 else {}
    cent_scores = list(cent_dict.values())
    q25, q75 = np.percentile(cent_scores, [25, 75]) if len(cent_scores) > 1 else [0.0, 0.0]
    for node in vis_G.nodes():
        role = roles.get(node, "inactive")
        node_cent = cent_dict.get(node, 0.0)
        tier_idx = 0 if node_cent <= q25 else (1 if node_cent < q75 else 2)
        chosen_color = HYBRID_PALETTES[role][tier_idx]
        deg = vis_G.degree(node)
        border_width = "1.0"
        if deg > 5:  border_width = "2.5"
        if deg > 15: border_width = "4.0"
        shape_type = "ellipse" if role == "inactive" else (
            "hexagon" if node_cent > q75 and role == "relay" else "box")
        vis_G.nodes[node].update({
            "style": "filled", "fillcolor": chosen_color,
            "color": "#111111" if tier_idx < 2 else "#000000",
            "fontcolor": "black" if tier_idx < 2 else "white",
            "shape": shape_type, "penwidth": border_width
        })
    return vis_G
def main():
    start_total = time.time()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== Multi-Threshold Heuristic-Labeled Topology Evaluation ===\n\n")

        for file1, file2, size_label in PAIRS:
            base_G1, times1 = load_and_parse_dot(os.path.join(FOLDER, file1))
            base_G2, times2 = load_and_parse_dot(os.path.join(FOLDER, file2))
            f.write(f"\n{'#'*60}\n")
            f.write(f"  GRAPH SIZE: {size_label}  |  {file1} vs {file2}\n")
            f.write(f"  Nodes G1: {base_G1.number_of_nodes()}  Edges G1: {base_G1.number_of_edges()}\n")
            f.write(f"  Nodes G2: {base_G2.number_of_nodes()}  Edges G2: {base_G2.number_of_edges()}\n")
            f.write(f"{'#'*60}\n\n")
            for thr in THRESHOLDS:
                start_thr = time.time()
                filter1, roles1 = get_valid_relay_nodes_and_roles(base_G1, times1, thr)
                filter2, roles2 = get_valid_relay_nodes_and_roles(base_G2, times2, thr)
                m1 = get_topology_metrics(filter1)
                m2 = get_topology_metrics(filter2)
                f.write(f"--- {size_label} | Temporal Relay <= {thr}h ---\n")
                if m1 and m2:
                    deg_s, clus_s, cent_s, path_s, relay_s, overall_s = compare_metrics(m1, m2)
                    elapsed = time.time() - start_thr
                    f.write(f"  Edges G1: {m1['edge_count']}  Nodes G1: {m1['node_count']}\n")
                    f.write(f"  Edges G2: {m2['edge_count']}  Nodes G2: {m2['node_count']}\n")
                    f.write(f"  Degree Alignment:      {deg_s:.4f}\n")
                    f.write(f"  Clustering Alignment:  {clus_s:.4f}\n")
                    f.write(f"  Centrality Alignment:  {cent_s:.4f}\n")
                    f.write(f"  Path Length Alignment: {path_s:.4f}\n")
                    f.write(f"  Relay Alignment:       {relay_s:.4f}\n")
                    f.write(f"  >>> Overall Similarity: {overall_s:.4f}\n")
                    f.write(f"  Time: {elapsed:.5f} s\n")
                    if size_label == "Small":
                        vis1 = apply_visualization(base_G1, filter1, roles1)
                        vis2 = apply_visualization(base_G2, filter2, roles2)
                        nx.drawing.nx_pydot.write_dot(
                            vis1, os.path.join(FOLDER, f"graph1_hybrid_{size_label}_thr_{thr}h.dot"))
                        nx.drawing.nx_pydot.write_dot(
                            vis2, os.path.join(FOLDER, f"graph2_hybrid_{size_label}_thr_{thr}h.dot"))
                else:
                    f.write("  Insufficient active temporal relay paths for this interval.\n")

                f.write("\n" + "=" * 60 + "\n\n")




if __name__ == "__main__":
    main()