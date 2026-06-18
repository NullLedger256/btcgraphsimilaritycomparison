import os
import re
import time
import hashlib
import networkx as nx
import numpy as np

FOLDER = r"x" #replace with actual path

PAIRS = [
    ("graph1smalllicit.dot",      "graph2smallilicit.dot",   "Small"),
    ("graph1medium1licit.dot", "graph2medium1ilicit.dot", "Medium"),
    ("graph1biglicit.dot",       "graph2bigilicit.dot",         "Large"),
]

OUTPUT_FILE = os.path.join(FOLDER, "x.txt") #replace with actual output file name
WL_ITERATIONS = 3

def load_dot_graph(filepath):
    G_raw   = nx.drawing.nx_pydot.read_dot(filepath)
    G_clean = nx.DiGraph()
    for u, v in nx.DiGraph(G_raw).edges():
        G_clean.add_edge(str(u).strip('"'), str(v).strip('"'))
    for node in G_raw.nodes():
        node_clean = str(node).strip('"')
        G_clean.add_node(node_clean, shape="box", fontsize="10")
        raw_label = G_raw.nodes[node].get("label", "")
        raw_label = " ".join(raw_label) if isinstance(raw_label, list) else str(raw_label)
        raw_label = raw_label.strip('"')
        val = re.search(r'Value:\s*([\d\.]+)\s*BTC', raw_label)
        tm  = re.search(r'Time:\s*([\d\-:\s]+(?:UTC|GMT)?)', raw_label)
        lbl = node_clean
        if val: lbl += f"\\nValue: {val.group(1)} BTC"
        if tm:  lbl += f"\\nTime: {tm.group(1).strip()}"
        G_clean.nodes[node_clean]["label"] = f'"{lbl}"'

    return G_clean


def run_directed_wl_pipeline(G1, G2, n_iter):
    PALETTE = [  #colors can be changed
        "lightgreen", "lightblue", "gold", "orchid", "orange",
        "turquoise", "pink", "khaki", "yellowgreen", "coral"
    ]

    labels_1 = {n: "1" for n in G1.nodes()}
    labels_2 = {n: "1" for n in G2.nodes()}

    for _ in range(n_iter):
        snap_1 = labels_1.copy()
        snap_2 = labels_2.copy()
        labels_1 = {
            n: hashlib.md5(
                f"{snap_1[n]}->[{','.join(sorted(snap_1[s] for s in G1.successors(n)))}]".encode()
            ).hexdigest()
            for n in G1.nodes()
        }
        labels_2 = {
            n: hashlib.md5(
                f"{snap_2[n]}->[{','.join(sorted(snap_2[s] for s in G2.successors(n)))}]".encode()
            ).hexdigest()
            for n in G2.nodes()
        }
    all_sigs  = sorted(set(labels_1.values()) | set(labels_2.values()))
    sig_index = {sig: idx for idx, sig in enumerate(all_sigs)}
    sig_color = {sig: PALETTE[idx % len(PALETTE)] for sig, idx in sig_index.items()}
    v1 = np.zeros(len(all_sigs))
    v2 = np.zeros(len(all_sigs))
    for sig in labels_1.values():
        v1[sig_index[sig]] += 1
    for sig in labels_2.values():
        v2[sig_index[sig]] += 1
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    cosine_score = (
        float(np.dot(v1, v2) / (norm1 * norm2))
        if norm1 > 0 and norm2 > 0 else 0.0
    )
    for G, labels in [(G1, labels_1), (G2, labels_2)]:
        for n in G.nodes():
            sig   = labels[n]
            in_d  = G.in_degree(n)
            out_d = G.out_degree(n)
            penwidth = "2.5" if in_d == 0 else ("1.0" if out_d == 0 else "1.8")
            G.nodes[n].update({
                "style": "filled", "fillcolor": sig_color[sig],
                "color": "black", "fontcolor": "black",
                "shape": "box", "fontsize": "10",
                "penwidth": penwidth, "comment": f"WL_Sig: {sig}",
            })

    return cosine_score, G1, G2


def main():
    report_lines = [
        "=" * 60,
        "  Baseline WL Structural Comparison (Cosine Similarity)",
        f"  WL Iterations: {WL_ITERATIONS}",
        "=" * 60,
        f"  {'Graph':<10} {'Nodes G1':>9} {'Edges G1':>9} {'Nodes G2':>9} {'Edges G2':>9} {'Similarity':>11} {'Time(s)':>10}",
        "  " + "-" * 58,
    ]
    for file1, file2, size_label in PAIRS:
        G1 = load_dot_graph(os.path.join(FOLDER, file1))
        G2 = load_dot_graph(os.path.join(FOLDER, file2))
        t0 = time.perf_counter()
        cosine_score, G1_col, G2_col = run_directed_wl_pipeline(G1, G2, WL_ITERATIONS)
        elapsed = time.perf_counter() - t0

        report_lines.append(
            f"  {size_label:<10} {len(G1):>9} {G1.number_of_edges():>9} "
            f"{len(G2):>9} {G2.number_of_edges():>9} {cosine_score:>11.4f} {elapsed:>10.5f}"
        )
        print(f"  {size_label}: sim={cosine_score:.4f}  time={elapsed:.5f}s")
        if size_label == "Small":
            nx.drawing.nx_pydot.write_dot(
                G1_col, os.path.join(FOLDER, f"graph1_baseline_wl_{size_label}.dot"))
            nx.drawing.nx_pydot.write_dot(
                G2_col, os.path.join(FOLDER, f"graph2_baseline_wl_{size_label}.dot"))

    report_lines += ["  " + "-" * 58, ""]
    report = "\n".join(report_lines)
    print("\n" + report)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    main()