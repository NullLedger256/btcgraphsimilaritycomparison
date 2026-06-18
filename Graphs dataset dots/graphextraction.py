import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed



API_KEY ='xxxx'  #replace with key
ROOT_ADDR = 'xxxx' #replace with a wallet address
MAX_DEPTH = 0 #replace with an actual depth
REQ_DELAY = 0.2
OUT_DOT = "xxxx" #replace with dot filename

MAX_WORKERS = 10
def children_limit(depth):
    if depth == 0:
        return 5
    return 10
thread_local = threading.local()
def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "btc-dot-generator"})
        thread_local.session = s
    return thread_local.session
def safe_get(url):
    for _ in range(5):
        try:
            r = get_session().get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            time.sleep(REQ_DELAY)
        except Exception:
            time.sleep(REQ_DELAY)
    return None
def sat_to_btc(x):
    try:
        return float(x) / 1e8
    except Exception:
        return 0.0
def fetch_transactions(address):
    url = (
        f"https://api.blockchair.com/"
        f"bitcoin/dashboards/address/{address}?key={API_KEY}"
    )
    js = safe_get(url)
    if not js or "data" not in js:
        return []
    try:
        return js["data"][address]["transactions"] or []
    except Exception:
        try:
            k = next(iter(js["data"].keys()))
            return js["data"][k]["transactions"] or []
        except Exception:
            return []
def fetch_tx_data(txid):
    url = (
        f"https://api.blockchair.com/"
        f"bitcoin/dashboards/transaction/{txid}?key={API_KEY}"
    )
    js = safe_get(url)
    if not js or "data" not in js:
        return None
    try:
        return js["data"][txid]
    except Exception:
        try:
            return next(iter(js["data"].values()))
        except Exception:
            return None
def is_outgoing(txdata, parent):
    for inp in txdata.get("inputs", []):
        addr = inp.get("recipient") or inp.get("address")
        if addr == parent:
            return True
    return False
def has_outgoing_tx(address):
    txids = fetch_transactions(address)
    for txid in txids:
        txdata = fetch_tx_data(txid)
        if not txdata:
            continue
        if is_outgoing(txdata, address):
            return True
        time.sleep(REQ_DELAY)
    return False

def get_first_outgoing_time(address):
    txids = fetch_transactions(address)
    for txid in txids:
        txdata = fetch_tx_data(txid)
        if not txdata:
            continue
        if is_outgoing(txdata, address):
            t = txdata.get("transaction", {}).get("time", None)
            time.sleep(REQ_DELAY)
            return t  # None if missing, caller must check
    return None



def bootstrap_root(address):
    txids = fetch_transactions(address)
    for txid in txids:
        txdata = fetch_tx_data(txid)
        if not txdata:
            continue
        if not is_outgoing(txdata, address):
            continue
        t = txdata.get("transaction", {}).get("time", None)
        if not t:                          # GUARD 1: no unknown on root
            continue
        value = 0.0
        for out in txdata.get("outputs", []):
            addr = out.get("recipient") or out.get("address")
            if addr and addr != address:
                value = sat_to_btc(out.get("value", 0))
                break
        return t, value
    return None, 0.0




def process_node(parent, depth, parent_time, visited_snapshot):
    limit = children_limit(depth)
    results = []
    txids = fetch_transactions(parent)
    children_added = 0
    for txid in txids:
        if children_added >= limit:
            break
        txdata = fetch_tx_data(txid)
        if not txdata:
            continue
        if not is_outgoing(txdata, parent):
            continue
        parent_tx_time = txdata.get("transaction", {}).get("time", None)
        if not parent_tx_time:
            continue
        outputs = txdata.get("outputs", [])
        for out in outputs:
            if children_added >= limit:
                break
            child = out.get("recipient") or out.get("address")
            if not child or child == parent or child in visited_snapshot:
                continue
            if not has_outgoing_tx(child):
                continue
            value_btc = sat_to_btc(out.get("value", 0))
            child_time = get_first_outgoing_time(child)
            if not child_time:
                print(f"    [SKIP] {child} -> no timestamp")
                continue
            if child_time <= parent_tx_time:
                print(f"    [SKIP] {child} -> not forward "
                      f"({child_time} <= {parent_tx_time})")
                continue
            results.append((parent, child, value_btc, child_time, parent_tx_time))
            children_added += 1
        time.sleep(REQ_DELAY)
    return results




def main():
    print(f"Bootstrapping root: {ROOT_ADDR}")
    root_time, root_value = bootstrap_root(ROOT_ADDR)
    if not root_time:
        print("ERROR: root has no valid timestamp. Cannot proceed.")
        return
    print(f"  Root timestamp : {root_time}")
    print(f"  Root value     : {root_value:.8f} BTC")
    visited = set()
    visited.add(ROOT_ADDR)
    edges = []
    node_labels = {}
    total_nodes = 1
    node_labels[ROOT_ADDR] = (
        f"{ROOT_ADDR}\\n"
        f"Value: {root_value:.8f} BTC\\n"
        f"Time: {root_time} UTC"
    )
    current_level = [(ROOT_ADDR, root_time, 0)]
    while current_level:
        depth = current_level[0][2]
        if depth >= MAX_DEPTH:
            break
        print(f"\n{'='*60}")
        print(f"  LEVEL {depth} -> expanding {len(current_level)} node(s) "
              f"(max {children_limit(depth)} children each)")
        print(f"{'='*60}")
        next_level = []
        level_lock = threading.Lock()
        completed = [0]
        found_children = [0]
        total_parents = len(current_level)
        visited_snapshot = frozenset(visited)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_parent = {
                executor.submit(
                    process_node, addr, depth, ts, visited_snapshot
                ): addr
                for addr, ts, _ in current_level
            }
            for future in as_completed(future_to_parent):
                parent_addr = future_to_parent[future]
                try:
                    child_results = future.result()
                except Exception as e:
                    print(f"  [ERROR] {parent_addr}: {e}")
                    child_results = []
                with level_lock:
                    completed[0] += 1
                    for (parent, child, value_btc, child_time, parent_tx_time) in child_results:
                        if child in visited:
                            continue
                        visited.add(child)
                        edges.append((parent, child))
                        node_labels[child] = (
                            f"{child}\\n"
                            f"Value: {value_btc:.8f} BTC\\n"
                            f"Time: {child_time} UTC"
                        )
                        next_level.append((child, child_time, depth + 1))
                        total_nodes += 1
                        found_children[0] += 1
                    print(
                        f"  [Lvl {depth}] {completed[0]}/{total_parents} done | "
                        f"{found_children[0]} children found | "
                        f"last: {parent_addr}"
                    )
        print(f"\n  LEVEL {depth} COMPLETE: "
              f"{len(current_level)} nodes expanded -> "
              f"{found_children[0]} children added (now at lvl {depth + 1})")

        current_level = next_level
    with open(OUT_DOT, "w", encoding="utf-8") as f:

        f.write("digraph G {\n")
        f.write("  rankdir=TB;\n")
        f.write('  node [shape=box, fontsize=10];\n\n')
        for parent, child in edges:
            f.write(f'  "{parent}" -> "{child}" [label=""];\n')
        f.write("\n")
        for node, label in node_labels.items():
            f.write(f'  "{node}" [label="{label}"];\n')
        f.write("}\n")
    print(f"\n{'='*60}")
    print(f"  DOT graph saved : {OUT_DOT}")
    print(f"  Total nodes     : {total_nodes}")
    print(f"  Total edges     : {len(edges)}")
    print(f"{'='*60}")
if __name__ == "__main__":
    main()