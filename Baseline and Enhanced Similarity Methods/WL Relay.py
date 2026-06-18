import os
import re
import time
import hashlib
import networkx as nx
import pandas as pd
import numpy as np


FOLDER = r"x" #replace with actual path

PAIRS = [
    ("graph1smalllicit.dot",      "graph2smallilicit.dot",   "Small"),
    ("graph1medium1licit.dot", "graph2medium1ilicit.dot", "Medium"),
    ("graph1biglicit.dot",       "graph2bigilicit.dot",         "Large"),
]

THRESHOLDS = [6, 24, 48, 72, 96]
WL_ITERATIONS = 3
OUTPUT_FILE = os.path.join(FOLDER, "x.txt") #replace with actual output file name


def load_and_parse_dot(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return nx.DiGraph(), {}

    print(f"Loading {os.path.basename(filepath)}...")
    G_raw = nx.drawing.nx_pydot.read_dot(filepath)
    G_clean = nx.DiGraph()
    outgoing_times = {}

    for u, v in nx.DiGraph(G_raw).edges():
        G_clean.add_edge(str(u).strip('"'), str(v).strip('"'))

    for node in G_raw.nodes():
        node_clean = str(node).strip('"')
        G_clean.add_node(node_clean, shape="box", fontsize="10")

        raw_label = G_raw.nodes[node].get("label", "")
        if isinstance(raw_label, list):
            raw_label = " ".join(raw_label)
        else:
            raw_label = str(raw_label)
        raw_label = raw_label.strip('"')

        val = re.search(r'Value:\s*([\d\.]+)\s*BTC', raw_label)
        tm  = re.search(r'Time:\s*([\d\-:\s]+(?:UTC|GMT)?)', raw_label)

        lbl = node_clean
        if val:
            lbl += f"\\nValue: {val.group(1)} BTC"
        if tm:
            lbl += f"\\nTimes: {tm.group(1).strip()}"

        G_clean.nodes[node_clean]["label"] = f'"{lbl}"'

        if tm:
            try:
                outgoing_times[node_clean] = pd.to_datetime(tm.group(1))
            except Exception:
                outgoing_times[node_clean] = pd.NaT
        else:
            outgoing_times[node_clean] = pd.NaT

    return G_clean, outgoing_times


def get_valid_relay_nodes(G, outgoing_times, max_hours):
    relay_centers = set()
    candidate_nodes = [
        n for n in G.nodes()
        if G.in_degree(n) > 0 and G.out_degree(n) > 0
    ]

    for v in candidate_nodes:
        t_v = outgoing_times.get(v)
        if t_v is None or pd.isna(t_v):
            continue

        preds = list(G.predecessors(v))
        succs = list(G.successors(v))
        relay_found = False

        for u in preds:
            t_u = outgoing_times.get(u)
            if t_u is None or pd.isna(t_u) or t_v < t_u:
                continue
            for w in succs:
                t_w = outgoing_times.get(w)
                if t_w is None or pd.isna(t_w) or t_w < t_v:
                    continue
                delta_hours = (t_w - t_v).total_seconds() / 3600.0
                if 0 <= delta_hours <= max_hours:
                    relay_centers.add(v)
                    relay_found = True
                    break
            if relay_found:
                break

    return relay_centers, relay_centers


def wl_hash(own_label, neighbor_labels):
    raw = own_label + "->[" + ",".join(sorted(neighbor_labels)) + "]"
    prefix = "active" if own_label.startswith("active") else "delayed"
    return f"{prefix}_{hashlib.md5(raw.encode()).hexdigest()[:12]}"


def run_wl_pipeline(G1, G2, relay_centers_1, relay_centers_2, n_iter):
    PALETTE = [  #colors can be changed
        "lightgreen", "lightblue", "gold", "orchid", "orange",
        "turquoise", "pink", "khaki", "yellowgreen", "coral"
    ]
    DELAYED_COLOR = "gray85"
    labels_1 = {n: ("active" if n in relay_centers_1 else "delayed") for n in G1.nodes()}
    labels_2 = {n: ("active" if n in relay_centers_2 else "delayed") for n in G2.nodes()}

    for _ in range(n_iter):
        snap_1 = labels_1.copy()
        snap_2 = labels_2.copy()
        labels_1 = {
            n: wl_hash(snap_1[n], [snap_1[s] for s in G1.successors(n)])
            for n in G1.nodes()
        }
        labels_2 = {
            n: wl_hash(snap_2[n], [snap_2[s] for s in G2.successors(n)])
            for n in G2.nodes()
        }
    all_sigs  = sorted(set(labels_1.values()) | set(labels_2.values()))
    sig_index = {sig: idx for idx, sig in enumerate(all_sigs)}
    active_idx = 0
    sig_color  = {}
    for sig in all_sigs:
        if sig.startswith("active"):
            sig_color[sig] = PALETTE[active_idx % len(PALETTE)]
            active_idx += 1
        else:
            sig_color[sig] = DELAYED_COLOR
    v1 = np.zeros(len(all_sigs))
    v2 = np.zeros(len(all_sigs))
    for sig in labels_1.values():
        v1[sig_index[sig]] += 1
    for sig in labels_2.values():
        v2[sig_index[sig]] += 1
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    cosine_score = (
        np.dot(v1, v2) / (norm1 * norm2)
        if (norm1 > 0 and norm2 > 0) else 0.0
    )

    def style_graph(G, labels):
        G_styled = G.copy()
        for n in G_styled.nodes():
            sig   = labels[n]
            in_d  = G_styled.in_degree(n)
            out_d = G_styled.out_degree(n)
            penwidth = "2.5" if in_d == 0 else ("1.0" if out_d == 0 else "1.8")
            G_styled.nodes[n].update({
                "style": "filled", "fillcolor": sig_color[sig],
                "shape": "box", "color": "black", "fontcolor": "black",
                "penwidth": penwidth, "margin": '"0.08,0.04"',
                "fontsize": "10", "comment": f"WL_sig: {sig}",
            })
        return G_styled

    return cosine_score, style_graph(G1, labels_1), style_graph(G2, labels_2)

def main():
    start_total = time.perf_counter()
    out_lines   = [
        "=== Multi-Threshold Temporal Relay WL Evaluation (Cosine Similarity) ===",
    ]
    for file1, file2, size_label in PAIRS:
        base_G1, out_times1 = load_and_parse_dot(os.path.join(FOLDER, file1))
        base_G2, out_times2 = load_and_parse_dot(os.path.join(FOLDER, file2))
        if len(base_G1) == 0 or len(base_G2) == 0:
            out_lines.append(f"\n[{size_label}] Error: graph loaded empty.\n")
            continue
        out_lines += [
            f"\n{'#'*55}",
            f"  GRAPH SIZE: {size_label}  |  {file1} vs {file2}",
            f"  G1: {base_G1.number_of_nodes()} nodes, {base_G1.number_of_edges()} edges",
            f"  G2: {base_G2.number_of_nodes()} nodes, {base_G2.number_of_edges()} edges",
            f"{'#'*55}",
        ]
        for th in THRESHOLDS:
            t0 = time.perf_counter()
            relay_centers_1, _ = get_valid_relay_nodes(base_G1, out_times1, th)
            relay_centers_2, _ = get_valid_relay_nodes(base_G2, out_times2, th)
            cosine_score, G1_styled, G2_styled = run_wl_pipeline(
                base_G1.copy(), base_G2.copy(),
                relay_centers_1, relay_centers_2,
                WL_ITERATIONS
            )
            if size_label == "Small":
                nx.drawing.nx_pydot.write_dot(
                    G1_styled, os.path.join(FOLDER, f"graph1_wl_relay_{size_label}_{th}h.dot"))
                nx.drawing.nx_pydot.write_dot(
                    G2_styled, os.path.join(FOLDER, f"graph2_wl_relay_{size_label}_{th}h.dot"))
            elapsed = time.perf_counter() - t0
            out_lines.append(
                f"  Threshold <= {th}h:\n"
                f"    WL Cosine Similarity:         {cosine_score:.4f}\n"
                f"    Relay-active nodes (G1 / G2): {len(relay_centers_1)} / {len(relay_centers_2)}\n"
                f"    Execution time:               {elapsed:.5f} s\n"
            )
            print(f"  {size_label} <= {th}h  sim={cosine_score:.4f}  "
                  f"active={len(relay_centers_1)}/{len(relay_centers_2)}  time={elapsed:.5f}s")
    out_lines.append(f"\nTotal runtime: {time.perf_counter() - start_total:.5f} s")
    final_output = "\n".join(out_lines)
    print("\n" + final_output)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_output)
if __name__ == "__main__":
    main()