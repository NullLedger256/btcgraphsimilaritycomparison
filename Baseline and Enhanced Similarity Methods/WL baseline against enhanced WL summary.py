import os, re, time, hashlib, itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

FOLDER = "."

LEGIT_FILES = [
    "graph1medium1licit.dot",
    "graph1lmedium3licit.dot",
    "graph1lmedium2licit.dot",
]
FOG_FILES = [
    "graph2medium1ilicit.dot",
    "graph2mediu2ilicit.dot",
    "graph2medium3ilicit.dot",
]

OUTPUT_PAPER  = os.path.join(FOLDER, "WL_summary_paper.txt")
OUTPUT_GITHUB = os.path.join(FOLDER, "WL_summary_github.txt")

WL_ITERATIONS    = 3
RELAY_THRESHOLDS = [6, 24, 48, 72, 96]
BTC_THRESHOLDS      = [0.00000001, 0.0001, 0.001, 0.01]
BTC_THRESHOLDS_STR  = ["0.00000001", "0.0001", "0.001", "0.01"]

TEMPORAL_BINS = [
    (1,            "ultra_dense"),
    (6,            "dense"),
    (24,           "moderate"),
    (float("inf"), "sparse"),
]

CATEGORIES = ["Ordinary–Ordinary", "Mixer–Mixer", "Ordinary–Mixer"]

def load_graph(filepath):
    G_raw = nx.drawing.nx_pydot.read_dot(filepath)
    G     = nx.DiGraph()
    for u, v in nx.DiGraph(G_raw).edges():
        G.add_edge(str(u).strip('"'), str(v).strip('"'))
    outgoing_times, values = {}, {}
    for node in G_raw.nodes():
        nc  = str(node).strip('"')
        G.add_node(nc)
        raw = G_raw.nodes[node].get("label", "")
        raw = " ".join(raw) if isinstance(raw, list) else str(raw)
        raw = raw.strip('"')
        val = re.search(r'Value:\s*([\d\.eE\+\-]+)\s*BTC', raw)
        tm  = re.search(r'Time:\s*([\d\-:\s]+(?:UTC|GMT)?)', raw)
        values[nc] = float(val.group(1)) if val else None
        try:
            outgoing_times[nc] = pd.to_datetime(tm.group(1)) if tm else pd.NaT
        except Exception:
            outgoing_times[nc] = pd.NaT
    if not nx.is_directed_acyclic_graph(G):
        while True:
            try:
                cycle = nx.find_cycle(G)
                G.remove_edge(cycle[-1][0], cycle[-1][1])
            except nx.NetworkXNoCycle:
                break
    return G, outgoing_times, values

def baseline_wl(G1, G2, n_iter=WL_ITERATIONS):
    l1 = {n: "1" for n in G1.nodes()}
    l2 = {n: "1" for n in G2.nodes()}
    for _ in range(n_iter):
        s1, s2 = l1.copy(), l2.copy()
        l1 = {n: hashlib.md5(f"{s1[n]}->[{','.join(sorted(s1[x] for x in G1.successors(n)))}]".encode()).hexdigest() for n in G1.nodes()}
        l2 = {n: hashlib.md5(f"{s2[n]}->[{','.join(sorted(s2[x] for x in G2.successors(n)))}]".encode()).hexdigest() for n in G2.nodes()}
    return _cosine_labels(l1, l2)

def _relay_centers(G, ot, max_hours):
    centers = set()
    for v in [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]:
        t_v = ot.get(v)
        if t_v is None or pd.isna(t_v):
            continue
        for u in G.predecessors(v):
            t_u = ot.get(u)
            if t_u is None or pd.isna(t_u) or t_v < t_u:
                continue
            for w in G.successors(v):
                t_w = ot.get(w)
                if t_w is None or pd.isna(t_w) or t_w < t_v:
                    continue
                if 0 <= (t_w - t_v).total_seconds() / 3600.0 <= max_hours:
                    centers.add(v)
                    break
            if v in centers:
                break
    return centers

def relay_wl_at(G1, G2, ot1, ot2, th, n_iter=WL_ITERATIONS):
    rc1 = _relay_centers(G1, ot1, th)
    rc2 = _relay_centers(G2, ot2, th)
    l1 = {n: ("active" if n in rc1 else "delayed") for n in G1.nodes()}
    l2 = {n: ("active" if n in rc2 else "delayed") for n in G2.nodes()}
    def _h(own, nbrs):
        raw = own + "->[" + ",".join(sorted(nbrs)) + "]"
        pfx = "active" if own.startswith("active") else "delayed"
        return f"{pfx}_{hashlib.md5(raw.encode()).hexdigest()[:12]}"
    for _ in range(n_iter):
        s1, s2 = l1.copy(), l2.copy()
        l1 = {n: _h(s1[n], [s1[x] for x in G1.successors(n)]) for n in G1.nodes()}
        l2 = {n: _h(s2[n], [s2[x] for x in G2.successors(n)]) for n in G2.nodes()}
    return _cosine_labels(l1, l2)

def _tbin(hours):
    for thr, lbl in TEMPORAL_BINS:
        if hours <= thr:
            return lbl
    return "sparse"

def _neigh_seed(n, G, ts):
    t_n  = ts.get(n)
    nbrs = list(G.predecessors(n)) + list(G.successors(n))
    if t_n is None or pd.isna(t_n) or not nbrs:
        return "T_none"
    diffs = [abs((ts[nb] - t_n).total_seconds()) / 3600.0
             for nb in nbrs if ts.get(nb) is not None and not pd.isna(ts.get(nb))]
    if not diffs:
        return "T_none"
    counts = defaultdict(int)
    for d in diffs:
        counts[_tbin(d)] += 1
    bin_names = [lbl for _, lbl in TEMPORAL_BINS]
    return "T_" + "_".join(f"{name[0].upper()}{counts.get(name,0)}" for name in bin_names)

def _wl_topo(G, init, n_iter):
    nodes = list(nx.topological_sort(G))
    n2i   = {n: i for i, n in enumerate(nodes)}
    cur   = dict(init)
    seqs  = {n2i[n]: [cur[n]] for n in G.nodes()}
    for _ in range(n_iter):
        new = {}
        for node in nodes:
            pl  = sorted(cur[p] for p in G.predecessors(node))
            sl  = sorted(cur[s] for s in G.successors(node))
            agg = f"{cur[node]}|P:{','.join(pl)}|C:{','.join(sl)}"
            new[node] = hashlib.md5(agg.encode()).hexdigest()[:8]
        cur = new
        for node in G.nodes():
            seqs[n2i[node]].append(cur[node])
    return seqs

def _fv(seqs):
    c = defaultdict(int)
    for seq in seqs.values():
        for i, lbl in enumerate(seq):
            c[f"i{i}_{lbl}"] += 1
    return c

def neighborhood_wl(G1, G2, ot1, ot2, n_iter=WL_ITERATIONS):
    s1 = {n: _neigh_seed(n, G1, ot1) for n in G1.nodes()}
    s2 = {n: _neigh_seed(n, G2, ot2) for n in G2.nodes()}
    return _cosine_dicts(_fv(_wl_topo(G1, s1, n_iter)), _fv(_wl_topo(G2, s2, n_iter)))

def _split_seed(n, G, vals, btc_thr):
    succs = list(G.successors(n))
    if not succs:
        return "SR_none"
    out_vals = [vals.get(s) for s in succs if vals.get(s) is not None]
    if not out_vals:
        return "SR_none"
    ratio = sum(1 for v in out_vals if v <= btc_thr) / len(succs)
    if ratio == 0.0:    return "SR_R0"
    elif ratio <= 0.25: return "SR_R25"
    elif ratio <= 0.5:  return "SR_R50"
    elif ratio <= 0.75: return "SR_R75"
    else:               return "SR_R100"

def split_wl_at(G1, G2, v1, v2, btc_thr, n_iter=WL_ITERATIONS):
    s1 = {n: _split_seed(n, G1, v1, btc_thr) for n in G1.nodes()}
    s2 = {n: _split_seed(n, G2, v2, btc_thr) for n in G2.nodes()}
    return _cosine_dicts(_fv(_wl_topo(G1, s1, n_iter)), _fv(_wl_topo(G2, s2, n_iter)))

def _cosine_labels(l1, l2):
    sigs = sorted(set(l1.values()) | set(l2.values()))
    idx  = {s: i for i, s in enumerate(sigs)}
    v1   = np.zeros(len(sigs)); v2 = np.zeros(len(sigs))
    for s in l1.values(): v1[idx[s]] += 1
    for s in l2.values(): v2[idx[s]] += 1
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0

def _cosine_dicts(d1, d2):
    keys = set(d1) | set(d2)
    a = np.array([d1.get(k, 0) for k in keys], dtype=float)
    b = np.array([d2.get(k, 0) for k in keys], dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

def _avg(lst):
    return sum(lst) / len(lst) if lst else float("nan")

def _mean_std(lst):
    if not lst:
        return float("nan"), float("nan")
    arr = np.array(lst, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))

def _fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"

def compute_all(legit, fog):
    oo_pairs = list(itertools.combinations(range(3), 2))
    mm_pairs = list(itertools.combinations(range(3), 2))
    om_pairs = list(itertools.product(range(3), range(3)))
    pair_defs = {
        "Ordinary–Ordinary": (oo_pairs, legit, legit,
            [f"Legit-{i+1} vs Legit-{j+1}" for i,j in oo_pairs]),
        "Mixer–Mixer":       (mm_pairs, fog,   fog,
            [f"Fog-{i+1} vs Fog-{j+1}"   for i,j in mm_pairs]),
        "Ordinary–Mixer":    (om_pairs, legit, fog,
            [f"Legit-{i+1} vs Fog-{j+1}" for i,j in om_pairs]),
    }
    scores = {}
    for cat, (pairs, pool_a, pool_b, labels) in pair_defs.items():
        scores[cat] = {}
        for (i, j), lbl in zip(pairs, labels):
            G_a, ot_a, v_a = pool_a[i]
            G_b, ot_b, v_b = pool_b[j]
            entry = {}
            t0 = time.perf_counter()
            entry["baseline"] = baseline_wl(G_a, G_b)
            entry["baseline_time"] = time.perf_counter() - t0
            entry["relay"] = {}
            entry["relay_time"] = {}
            for th in RELAY_THRESHOLDS:
                t0 = time.perf_counter()
                s = relay_wl_at(G_a, G_b, ot_a, ot_b, th)
                entry["relay"][th] = s
                entry["relay_time"][th] = time.perf_counter() - t0
            t0 = time.perf_counter()
            entry["neigh"] = neighborhood_wl(G_a, G_b, ot_a, ot_b)
            entry["neigh_time"] = time.perf_counter() - t0
            entry["split"] = {}
            entry["split_time"] = {}
            for bthr, bstr in zip(BTC_THRESHOLDS, BTC_THRESHOLDS_STR):
                t0 = time.perf_counter()
                s = split_wl_at(G_a, G_b, v_a, v_b, bthr)
                entry["split"][bthr] = s
                entry["split_time"][bthr] = time.perf_counter() - t0
            scores[cat][lbl] = entry
    return scores

def build_paper_table(scores, runtime):
    lines = []
    W = 110
    lines += [
        "=" * W,
        "  TABLE — WL Cosine Similarity  (mean ± std over graph pairs in category)",
        "  Ordinary = legitimate graphs  |  Mixer = fog/mixing graphs",
        "=" * W,
    ]
    col_w = 18
    lines.append(
        f"  {'Pair Category':<20} "
        f"{'Baseline':>{col_w}}  "
        f"{'Relay WL':>{col_w}}  "
        f"{'Neighborhood':>{col_w}}  "
        f"{'Fragmentation Split WL':>{col_w}}  "
        f"{'Avg Time(s)':>12}"
    )
    lines.append("  " + "-" * (W - 2))
    for cat in CATEGORIES:
        cat_data = scores[cat]
        base_vals  = [v["baseline"]                   for v in cat_data.values()]
        relay_vals = [_avg(list(v["relay"].values())) for v in cat_data.values()]
        neigh_vals = [v["neigh"]                      for v in cat_data.values()]
        split_vals = [_avg(list(v["split"].values())) for v in cat_data.values()]
        pair_times = []
        for v in cat_data.values():
            t = (v["baseline_time"] + v["neigh_time"] + sum(v["relay_time"].values()) + sum(v["split_time"].values()))
            pair_times.append(t)
        avg_time = _avg(pair_times)
        b_m,  b_s  = _mean_std(base_vals)
        r_m,  r_s  = _mean_std(relay_vals)
        n_m,  n_s  = _mean_std(neigh_vals)
        sp_m, sp_s = _mean_std(split_vals)
        lines.append(
            f"  {cat:<20} "
            f"{_fmt(b_m,  b_s):>{col_w}}  "
            f"{_fmt(r_m,  r_s):>{col_w}}  "
            f"{_fmt(n_m,  n_s):>{col_w}}  "
            f"{_fmt(sp_m, sp_s):>{col_w}}  "
            f"{avg_time:>12.5f}"
        )
    lines += [
        "  " + "-" * (W - 2),
        f"  • WL iterations = {WL_ITERATIONS}  |  Total runtime: {runtime:.2f} s",
        "=" * W,
    ]
    return "\n".join(lines)

def build_github_table(scores, runtime):
    lines = []
    W = 100
    lines.append("=" * W)
    lines.append("  WL COSINE SIMILARITY: DETAILED PER-THRESHOLD REPORT")
    lines.append("=" * W)
    for cat in CATEGORIES:
        cat_data = scores[cat]
        lines += ["", f"  ── {cat} ──", ""]
        lines.append(f"  {'Pair':<22} {'Method':<22} {'Score':>8}  {'Threshold':>14}  {'Time(s)':>10}")
        lines.append("  " + "-" * 82)
        for lbl, entry in cat_data.items():
            lines.append(f"  {lbl:<22} {'Baseline':<22} {entry['baseline']:>8.4f}  {'—':>14}  {entry['baseline_time']:>10.5f}")
            lines.append(f"  {'':22} {'Neighborhood':<22} {entry['neigh']:>8.4f}  {'—':>14}  {entry['neigh_time']:>10.5f}")
            for th, s in entry["relay"].items():
                lines.append(f"  {'':22} {'Relay WL':<22} {s:>8.4f}  {f'δ≤{th}h':>14}  {entry['relay_time'][th]:>10.5f}")
            for bthr, bstr in zip(BTC_THRESHOLDS, BTC_THRESHOLDS_STR):
                s = entry["split"][bthr]
                lines.append(f"  {'':22} {'Fragmentation Split WL':<22} {s:>8.4f}  {f'≤{bstr} BTC':>16}  {entry['split_time'][bthr]:>10.5f}")
            lines.append("")
    lines += [
        "=" * W,
        f"  WL iterations = {WL_ITERATIONS}  |  Total runtime: {runtime:.2f} s",
        "=" * W,
    ]
    return "\n".join(lines)

def main():
    t_start = time.perf_counter()
    legit = [load_graph(os.path.join(FOLDER, f)) for f in LEGIT_FILES]
    fog   = [load_graph(os.path.join(FOLDER, f)) for f in FOG_FILES]
    scores  = compute_all(legit, fog)
    runtime = time.perf_counter() - t_start
    paper_txt  = build_paper_table(scores, runtime)
    github_txt = build_github_table(scores, runtime)
    print(paper_txt)
    with open(OUTPUT_PAPER,  "w", encoding="utf-8") as f: f.write(paper_txt)
    with open(OUTPUT_GITHUB, "w", encoding="utf-8") as f: f.write(github_txt)
    print(f"\nSaved → {OUTPUT_PAPER} and {OUTPUT_GITHUB}")

if __name__ == "__main__":
    main()