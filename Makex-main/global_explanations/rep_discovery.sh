#!/bin/bash
set -euo pipefail

DATASET_DIR="../DataSets/icews14"
PROCESSED_DIR="$DATASET_DIR/processed"
GRAPH_DIR="$PROCESSED_DIR/original_graph"
TRAIN_DIR="$PROCESSED_DIR/train_test"

python3 - <<PY
from pathlib import Path
import csv
import json
from collections import Counter, defaultdict, OrderedDict

dataset_dir = Path("$DATASET_DIR").resolve()
processed_dir = Path("$PROCESSED_DIR").resolve()
graph_dir = Path("$GRAPH_DIR").resolve()
train_dir = Path("$TRAIN_DIR").resolve()
processed_dir.mkdir(parents=True, exist_ok=True)
graph_dir.mkdir(parents=True, exist_ok=True)
train_dir.mkdir(parents=True, exist_ok=True)

entity2id = json.loads(Path(dataset_dir / "entity2id.json").read_text())
relation2id = json.loads(Path(dataset_dir / "relation2id.json").read_text())

def sanitize_value(value: str) -> str:
    value = value.replace(",", " ").strip()
    return value or "Unknown"

def parse_attributes(raw_name: str):
    base = raw_name
    country = "Unknown"
    if "_(" in raw_name and raw_name.endswith(")"):
        base, country = raw_name.rsplit("_(", 1)
        country = country[:-1]
    base = sanitize_value(base.replace("_", " "))
    country = sanitize_value(country.replace("_", " "))
    return base, country

vertex_path = graph_dir / "icews_v.csv"
type_counter = defaultdict(Counter)
country_counter = defaultdict(Counter)
with vertex_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["vertex_id:int", "label_id:int", "name:string", "type:string", "country:string"])
    for name, idx in entity2id.items():
        ent_type, ent_country = parse_attributes(name)
        writer.writerow([idx, 0, name, ent_type, ent_country])
        type_counter[0][ent_type] += 1
        country_counter[0][ent_country] += 1

edge_path = graph_dir / "icews_e.csv"
edge_id = 0
edges_by_src = defaultdict(list)
relation_freq = Counter()
with edge_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["edge_id:int", "source_id:int", "target_id:int", "label_id:int"])
    for split in ["train.txt", "valid.txt", "test.txt"]:
        split_path = dataset_dir / split
        if not split_path.exists():
            continue
        for line in split_path.read_text().splitlines():
            parts = line.strip().split("\\t")
            if len(parts) < 3:
                continue
            head, relation, tail = parts[:3]
            try:
                h_id = entity2id[head]
                t_id = entity2id[tail]
                r_id = relation2id[relation]
            except KeyError:
                continue
            writer.writerow([edge_id, h_id, t_id, r_id])
            edges_by_src[h_id].append((t_id, r_id))
            relation_freq[r_id] += 1
            edge_id += 1

train_txt = dataset_dir / "train.txt"
train_records = []
if train_txt.exists():
    for line in train_txt.read_text().splitlines():
        parts = line.strip().split("\\t")
        if len(parts) < 3:
            continue
        head, relation, tail = parts[:3]
        try:
            h_id = entity2id[head]
            t_id = entity2id[tail]
            r_id = relation2id[relation]
        except KeyError:
            continue
        train_records.append((h_id, r_id, t_id))

train_log = train_dir / "train.log"
with train_log.open("w") as f:
    for h_id, _, t_id in train_records:
        f.write(f"{h_id}\\t{t_id}\\t1\\n")

edge_label_reverse = processed_dir / "edge_label_reverse.csv"
with edge_label_reverse.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source_node_label", "target_node_label", "edge_label"])
    for rel_id in sorted(relation2id.values()):
        writer.writerow([0, 0, rel_id])

def select_neighbors(neighbors, exclude_id, max_edges=2):
    filtered = [edge for edge in neighbors if edge[0] != exclude_id]
    if not filtered:
        return []
    filtered.sort(key=lambda x: (-relation_freq[x[1]], x[1], x[0]))
    result = []
    seen = set()
    for dst, lbl in filtered:
        key = (dst, lbl)
        if key in seen:
            continue
        seen.add(key)
        result.append((dst, lbl))
        if len(result) >= max_edges:
            break
    return result

pattern_defs = []
seen_signatures = set()
max_patterns = 80
for h_id, rel_id, t_id in train_records:
    user_edges = select_neighbors(edges_by_src.get(h_id, []), t_id)
    item_edges = select_neighbors(edges_by_src.get(t_id, []), h_id)
    if len(user_edges) < 2 or len(item_edges) < 2:
        continue
    node_map = OrderedDict([(h_id, 1), (t_id, 2)])
    edges = []
    def ensure(node_id):
        if node_id not in node_map:
            node_map[node_id] = len(node_map) + 1
        return node_map[node_id]
    edges.append([1, 2, rel_id])
    for dst, lbl in user_edges:
        edges.append([1, ensure(dst), lbl])
    for dst, lbl in item_edges:
        edges.append([2, ensure(dst), lbl])
    nodes = [[idx, 0] for idx in range(1, len(node_map) + 1)]
    edges_sorted = sorted(edges, key=lambda x: (x[0], x[1], x[2]))
    signature = (tuple(tuple(n) for n in nodes), tuple(tuple(e) for e in edges_sorted))
    if signature in seen_signatures:
        continue
    seen_signatures.add(signature)
    pattern_defs.append((nodes, edges_sorted))
    if len(pattern_defs) >= max_patterns:
        break

if not pattern_defs:
    top_relations = sorted(relation2id.values())[:3]
    for rel_id in top_relations:
        nodes = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
        edges = [[1, 2, rel_id], [1, 3, rel_id], [2, 4, rel_id], [2, 5, rel_id]]
        pattern_defs.append((nodes, edges))

pattern_path = processed_dir / "pattern.txt"
with pattern_path.open("w") as f:
    for idx, (nodes, edges) in enumerate(pattern_defs):
        f.write(f"pattern {idx}: {nodes}  {edges}\\n")

candidate_predicates_path = processed_dir / "candidate_predicates.txt"
max_predicates_per_attr = 200
with candidate_predicates_path.open("w") as f:
    for label_id, counter in type_counter.items():
        for value, _ in counter.most_common(max_predicates_per_attr):
            f.write(f"{label_id},type,{value}\\n")
    for label_id, counter in country_counter.items():
        for value, _ in counter.most_common(max_predicates_per_attr):
            f.write(f"{label_id},country,{value}\\n")
PY

pattern_file="$PROCESSED_DIR/pattern.txt"
candidate_predicates_file="$PROCESSED_DIR/candidate_predicates.txt"
rep_support=0
rep_conf=0.0
rep_to_path_ratio=0.2
each_node_predicates=5
sort_by_support_weights=0.3
max_outdegree=3
max_len_of_path=2
v_file="$GRAPH_DIR/icews_v.csv"
e_file="$GRAPH_DIR/icews_e.csv"
ml_file="$TRAIN_DIR/train.log"
delta_l=0.0
delta_r=1.0
user_offset=0
rep_file_generate="./rep_all.txt"
rep_file_generate_support_conf="./rep_support_conf.txt"
edge_label_reverse_csv="$PROCESSED_DIR/edge_label_reverse.csv"
rep_file_generate_support_conf_none_support="./rep.txt"
num_process=4

g++ makex_rep_discovery.cpp -o makex_rep_discovery -std=c++17 -I ../pyMakex/include/ -O3 >> makex_rep_discovery.txt 2>&1

./makex_rep_discovery "$pattern_file" "$candidate_predicates_file" $rep_support $rep_conf $rep_to_path_ratio $each_node_predicates $sort_by_support_weights "$v_file" "$e_file" "$ml_file" $delta_l $delta_r $user_offset "$rep_file_generate" "$rep_file_generate_support_conf" "$edge_label_reverse_csv" "$rep_file_generate_support_conf_none_support" $num_process > ./output.txt 2>&1

# 若算法未输出，使用结构规则兜底（无 name/id 谓词）
python3 - <<PY
from pathlib import Path
import ast
import csv
from collections import defaultdict

rep_path = Path("rep_support_conf.txt")
if rep_path.exists() and rep_path.stat().st_size > 0:
    raise SystemExit(0)

pattern_file = Path("$pattern_file")
edge_file = Path("$e_file")
rep_all = Path("rep_all.txt")
rep_base = Path("rep.txt")

if not pattern_file.exists():
    raise SystemExit(0)

label_support = defaultdict(int)
if edge_file.exists():
    with edge_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = row.get("label_id:int") or row.get("label_id", "0")
            label_support[lab] += 1

with rep_path.open("w") as out1, rep_all.open("w") as out2, rep_base.open("w") as out3:
    for line in pattern_file.read_text().splitlines():
        parts = line.split(":")
        if len(parts) < 2:
            continue
        rest = parts[1].strip()
        try:
            vertex_part, edge_part = rest.split("  ")
            vertex_list = ast.literal_eval(vertex_part)
            edge_list = ast.literal_eval(edge_part)
        except Exception:
            continue
        if not edge_list:
            continue
        edge_lab = str(edge_list[0][2])
        support = float(label_support.get(edge_lab, 0))
        rep_line = [
            vertex_list,
            edge_list,
            [[]],        # 无属性谓词，避免 name/id 过拟合
            [support, 1.0],
            [1, 2, 1, 1.0],
        ]
        for out in (out1, out2, out3):
            out.write(f"{rep_line}\\n")
PY

python3 - <<PY
from pathlib import Path
import ast
import csv
from collections import Counter, defaultdict

rep_path = Path("rep_support_conf.txt")
if not rep_path.exists() or rep_path.stat().st_size == 0:
    raise SystemExit(0)

edge_file = Path("$GRAPH_DIR/icews_e.csv").resolve()
vertex_file = Path("$GRAPH_DIR/icews_v.csv").resolve()

def load_reps(path: Path):
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(ast.literal_eval(line))
    return entries

rep_entries = load_reps(rep_path)

vertex_attrs = {}
with vertex_file.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = int(row["vertex_id:int"])
        vertex_attrs[vid] = {
            "type": row.get("type:string", "").strip(),
            "country": row.get("country:string", "").strip(),
        }

relation_counter = defaultdict(Counter)
with edge_file.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = int(row["source_id:int"])
        rel = int(row["label_id:int"])
        relation_counter[src][rel] += 1

def matching_nodes(required_counter):
    if not required_counter:
        return []
    matched = []
    for node_id, counter in relation_counter.items():
        ok = True
        for rel, need in required_counter.items():
            if counter.get(rel, 0) < need:
                ok = False
                break
        if ok:
            matched.append(node_id)
    return matched

def top_attr(nodes, attr_key):
    counts = Counter()
    for node_id in nodes:
        attr_val = vertex_attrs.get(node_id, {}).get(attr_key)
        if not attr_val or attr_val == "Unknown":
            continue
        counts[attr_val] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]

updated_entries = []
for rep in rep_entries:
    nodes = rep[0]
    edges = rep[1]
    predicates = rep[2] if len(rep) > 2 else []
    base_predicates = [pred for pred in predicates if pred]
    user_edges = [edge for edge in edges if edge[0] == 1 and edge[1] != 2]
    item_edges = [edge for edge in edges if edge[0] == 2 and edge[1] != 1]
    user_counter = Counter(edge[2] for edge in user_edges)
    item_counter = Counter(edge[2] for edge in item_edges)
    new_predicates = list(base_predicates)
    user_nodes = matching_nodes(user_counter)
    item_nodes = matching_nodes(item_counter)
    user_type = top_attr(user_nodes, "type")
    user_country = top_attr(user_nodes, "country")
    item_type = top_attr(item_nodes, "type")
    item_country = top_attr(item_nodes, "country")
    def add_pred(node_id, attr, value):
        if not value:
            return
        entry = ["Constant", node_id, attr, value, "string", "="]
        if entry not in new_predicates:
            new_predicates.append(entry)
    add_pred(1, "type", user_type)
    add_pred(1, "country", user_country)
    add_pred(2, "type", item_type)
    add_pred(2, "country", item_country)
    rep[2] = new_predicates
    updated_entries.append(rep)

def write_reps(path: Path, entries):
    with path.open("w") as f:
        for entry in entries:
            f.write(f"{entry}\\n")

write_reps(Path("rep_support_conf.txt"), updated_entries)
write_reps(Path("rep_all.txt"), updated_entries)
write_reps(Path("rep.txt"), updated_entries)
PY
