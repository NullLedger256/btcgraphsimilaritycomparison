import os, re, time, itertools
from collections import Counter, defaultdict
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cosine


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

OUTPUT_PAPER  = os.path.join(FOLDER, "top_summary_paper.txt")
OUTPUT_GITHUB = os.path.join(FOLDER, "top_summary_github.txt")

RELAY_THRESHOLDS    = [6, 24, 48, 72, 96]
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
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    timestamps, values = {}, {}
    label_pat = re.compile(r'"([^"]+)"\s*\[label="([^"]+)"\]')
    for node_id, label in label_pat.findall(content):
        tm  = re.search(r'Time:\s+([\d\-: ]+)', label)
        val = re.search(r'Value:\s*([\d\.eE\+\-]+)\s*BTC', label)
        try:
            timestamps[node_id] = pd.to_datetime(tm.group(1).strip()) if tm else pd.NaT
        except Exception:
            timestamps[node_id] = pd.NaT
        values[node_id] = float(val.group(1)) if val else None
    edge_pat = re.compile(r'"([^"]+)"\s*->\s*"([^"]+)"')
    edges    = edge_pat.findall(content)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    for node_id in timestamps:
        if node_id not in G:
            G.add_node(node_id)
    if not nx.is_directed_acyclic_graph(G):
        while True:
            try:
                cycle = nx.find_cycle(G)
                G.remove_edge(cycle[-1][0], cycle[-1][1])
            except nx.NetworkXNoCycle:
                break
    return G, timestamps, values


def _norm_hist(counter, max_val=20):
    vec   = np.array([counter.get(i, 0) for i in range(max_val + 1)], dtype=float)
    total = vec.sum()
    return vec / total if total > 0 else vec

def _safe_cosine(a, b):
    min_l = min(len(a), len(b))
    if min_l == 0:
        return 0.0
    return float(np.clip(1.0 - cosine(a[:min_l], b[:min_l]), 0.0, 1.0))

def _path_vec(G_und, max_val=10):
    if len(G_und) == 0:
        return np.zeros(max_val + 1)
    try:
        if not nx.is_connected(G_und):
            G_und = G_und.subgraph(max(nx.connected_components(G_und), key=len)).copy()
        lengths = [l for s in dict(nx.shortest_path_length(G_und)).values()
                   for l in s.values() if 0 < l <= max_val]
        return _norm_hist(Counter(lengths), max_val)
    except Exception:
        return np.zeros(max_val + 1)

def _base_metrics(G):
    G_und      = G.to_undirected()
    deg_hist   = Counter(d for _, d in G.degree())
    clustering = nx.average_clustering(G_und) if len(G_und) > 0 else 0.0
    centrality = sorted(nx.degree_centrality(G).values())
    pv         = _path_vec(G_und)
    return deg_hist, clustering, centrality, pv

def _compare_base(dh1, cl1, ce1, pv1, dh2, cl2, ce2, pv2):
    all_degs = set(dh1) | set(dh2)
    v1 = np.array([dh1.get(d, 0) for d in all_degs], dtype=float)
    v2 = np.array([dh2.get(d, 0) for d in all_degs], dtype=float)
    deg_score  = 1.0 - min(1.0, euclidean(v1 / (v1.sum() or 1), v2 / (v2.sum() or 1)))
    clust_score= 1.0 - abs(cl1 - cl2)
    cent_score = _safe_cosine(ce1, ce2)
    path_score = 1.0 - min(1.0, euclidean(pv1, pv2))
    return deg_score, clust_score, cent_score, path_score

def baseline_tfe(G1, G2):
    dh1, cl1, ce1, pv1 = _base_metrics(G1)
    dh2, cl2, ce2, pv2 = _base_metrics(G2)
    deg_s, clu_s, cen_s, pth_s = _compare_base(dh1, cl1, ce1, pv1, dh2, cl2, ce2, pv2)
    return (deg_s + clu_s + cen_s + pth_s) / 4.0

def _relay_edges(G, ts, max_hours):
    valid = set()
    for v in G.nodes():
        t_v = ts.get(v)
        if t_v is None or pd.isna(t_v):
            continue
        for u in G.predecessors(v):
            t_u = ts.get(u)
            if t_u is None or pd.isna(t_u) or t_v < t_u:
                continue
            for w in G.successors(v):
                t_w = ts.get(w)
                if t_w is None or pd.isna(t_w) or t_w < t_v:
                    continue
                if (t_w - t_v).total_seconds() / 3600.0 <= max_hours:
                    valid.add((u, v))
                    valid.add((v, w))
    return list(valid)

def relay_tfe_at(G1, G2, ts1, ts2, th):
    e1 = _relay_edges(G1, ts1, th)
    e2 = _relay_edges(G2, ts2, th)
    if not e1 or not e2:
        return 0.0
    sub1 = nx.DiGraph(); sub1.add_edges_from(e1)
    sub2 = nx.DiGraph(); sub2.add_edges_from(e2)
    dh1, cl1, ce1, pv1 = _base_metrics(sub1)
    dh2, cl2, ce2, pv2 = _base_metrics(sub2)
    deg_s, clu_s, cen_s, pth_s = _compare_base(dh1, cl1, ce1, pv1, dh2, cl2, ce2, pv2)
    rd1 = len(e1) / max(len(sub1), 1)
    rd2 = len(e2) / max(len(sub2), 1)
    relay_s = 1.0 - min(1.0, abs(rd1 - rd2))
    return (deg_s + clu_s + cen_s + pth_s + relay_s) / 5.0

def _tbin(hours):
    for thr, lbl in TEMPORAL_BINS:
        if hours <= thr:
            return lbl
    return "sparse"

def _neigh_sig(n, G, ts):
    t_n = ts.get(n)
    if t_n is None or pd.isna(t_n):
        return "none"
    diffs = []
    for nb in list(G.predecessors(n)) + list(G.successors(n)):
        t_nb = ts.get(nb)
        if t_nb is not None and not pd.isna(t_nb):
            diffs.append(abs((t_nb - t_n).total_seconds()) / 3600.0)
    if not diffs:
        return "none"
    counts = defaultdict(int)
    for d in diffs:
        counts[_tbin(d)] += 1
    bin_names = [lbl for _, lbl in TEMPORAL_BINS]
    return "T_" + "_".join(f"{name[0].upper()}{counts.get(name, 0)}" for name in bin_names)

def neighborhood_tfe(G1, G2, ts1, ts2):
    dh1, cl1, ce1, pv1 = _base_metrics(G1)
    dh2, cl2, ce2, pv2 = _base_metrics(G2)
    deg_s, clu_s, cen_s, pth_s = _compare_base(dh1, cl1, ce1, pv1, dh2, cl2, ce2, pv2)
    sig1 = Counter(_neigh_sig(n, G1, ts1) for n in G1.nodes())
    sig2 = Counter(_neigh_sig(n, G2, ts2) for n in G2.nodes())
    all_sigs = set(sig1) | set(sig2)
    sv1 = np.array([sig1.get(s, 0) for s in all_sigs], dtype=float)
    sv2 = np.array([sig2.get(s, 0) for s in all_sigs], dtype=float)
    neigh_s = 1.0 - min(1.0, euclidean(sv1 / (sv1.sum() or 1), sv2 / (sv2.sum() or 1)))
    return (deg_s + clu_s + cen_s + pth_s + neigh_s) / 5.0

def _split_label(n, G, vals, btc_thr):
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

def split_tfe_at(G1, G2, v1, v2, btc_thr):
    dh1, cl1, ce1, pv1 = _base_metrics(G1)
    dh2, cl2, ce2, pv2 = _base_metrics(G2)
    deg_s, clu_s, cen_s, pth_s = _compare_base(dh1, cl1, ce1, pv1, dh2, cl2, ce2, pv2)
    sd1 = Counter(_split_label(n, G1, v1, btc_thr) for n in G1.nodes())
    sd2 = Counter(_split_label(n, G2, v2, btc_thr) for n in G2.nodes())
    all_labs = set(sd1) | set(sd2)
    sv1 = np.array([sd1.get(l, 0) for l in all_labs], dtype=float)
    sv2 = np.array([sd2.get(l, 0) for l in all_labs], dtype=float)
    split_s = 1.0 - min(1.0, euclidean(sv1 / (sv1.sum() or 1), sv2 / (sv2.sum() or 1)))
    return (deg_s + clu_s + cen_s + pth_s + split_s) / 5.0

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
            [f"Legit-{i+1} vs Legit-{j+1}" for i, j in oo_pairs]),
        "Mixer–Mixer":       (mm_pairs, fog,   fog,
            [f"Fog-{i+1} vs Fog-{j+1}"     for i, j in mm_pairs]),
        "Ordinary–Mixer":    (om_pairs, legit, fog,
            [f"Legit-{i+1} vs Fog-{j+1}"   for i, j in om_pairs]),
    }
    scores = {}
    for cat, (pairs, pool_a, pool_b, labels) in pair_defs.items():
        scores[cat] = {}
        for (i, j), lbl in zip(pairs, labels):
            G_a, ts_a, v_a = pool_a[i]
            G_b, ts_b, v_b = pool_b[j]
            entry = {}
            t0 = time.perf_counter()
            entry["baseline"] = baseline_tfe(G_a, G_b)
            entry["baseline_time"] = time.perf_counter() - t0
            entry["relay"] = {}
            entry["relay_time"] = {}
            for th in RELAY_THRESHOLDS:
                t0 = time.perf_counter()
                s = relay_tfe_at(G_a, G_b, ts_a, ts_b, th)
                entry["relay"][th] = s
                entry["relay_time"][th] = time.perf_counter() - t0
            t0 = time.perf_counter()
            entry["neigh"] = neighborhood_tfe(G_a, G_b, ts_a, ts_b)
            entry["neigh_time"] = time.perf_counter() - t0
            entry["split"] = {}
            entry["split_time"] = {}
            for bthr, bstr in zip(BTC_THRESHOLDS, BTC_THRESHOLDS_STR):
                t0 = time.perf_counter()
                s = split_tfe_at(G_a, G_b, v_a, v_b, bthr)
                entry["split"][bthr] = s
                entry["split_time"][bthr] = time.perf_counter() - t0
            scores[cat][lbl] = entry
    return scores

def build_paper_table(scores, runtime):
    lines = []
    W = 110
    lines += [
        "=" * W,
        "  TABLE — Topological Feature Extraction Similarity  (mean ± std over graph pairs in category)",
        "  Ordinary = legitimate graphs  |  Mixer = fog/mixing graphs",
        "=" * W,
    ]
    col_w = 18
    lines.append(
        f"  {'Pair Category':<20} "
        f"{'Baseline':>{col_w}}  "
        f"{'Relay TFE':>{col_w}}  "
        f"{'Neighborhood':>{col_w}}  "
        f"{'Fragmentation Split TFE':>{col_w}}  "
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
        f"  • Total runtime: {runtime:.2f} s",
        "=" * W,
    ]
    return "\n".join(lines)

def build_github_table(scores, runtime):
    lines = []
    W = 100
    lines += ["=" * W, "  FULL PER-THRESHOLD BREAKDOWN — Topological Feature Extraction Similarity", "=" * W]
    for cat in CATEGORIES:
        cat_data = scores[cat]
        lines += ["", f"  ── {cat} ──", ""]
        lines.append(f"  {'Pair':<22} {'Method':<22} {'Score':>8}  {'Threshold':>16}  {'Time(s)':>10}")
        lines.append("  " + "-" * 84)
        for lbl, entry in cat_data.items():
            lines.append(f"  {lbl:<22} {'Baseline':<22} {entry['baseline']:>8.4f}  {'—':>16}  {entry['baseline_time']:>10.5f}")
            lines.append(f"  {'':22} {'Neighborhood':<22} {entry['neigh']:>8.4f}  {'—':>16}  {entry['neigh_time']:>10.5f}")
            for th, s in entry["relay"].items():
                lines.append(f"  {'':22} {'Relay TFE':<22} {s:>8.4f}  {f'δ≤{th}h':>16}  {entry['relay_time'][th]:>10.5f}")
            for bthr, bstr in zip(BTC_THRESHOLDS, BTC_THRESHOLDS_STR):
                s = entry["split"][bthr]
                lines.append(f"  {'':22} {'Fragmentation Split TFE':<22} {s:>8.4f}  {f'≤{bstr} BTC':>16}  {entry['split_time'][bthr]:>10.5f}")
            lines.append("")
    lines += ["=" * W, f"  Total runtime: {runtime:.2f} s", "=" * W]
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