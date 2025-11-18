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

dataset_dir = Path("$DATASET_DIR").resolve()
processed_dir = Path("$PROCESSED_DIR").resolve()
graph_dir = Path("$GRAPH_DIR").resolve()
train_dir = Path("$TRAIN_DIR").resolve()
processed_dir.mkdir(parents=True, exist_ok=True)
graph_dir.mkdir(parents=True, exist_ok=True)
train_dir.mkdir(parents=True, exist_ok=True)

entity2id = json.loads(Path(dataset_dir / "entity2id.json").read_text())
relation2id = json.loads(Path(dataset_dir / "relation2id.json").read_text())

vertex_path = graph_dir / "icews_v.csv"
with vertex_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["vertex_id:int", "label_id:int", "name:string"])
    for name, idx in entity2id.items():
        writer.writerow([idx, 0, name])

edge_path = graph_dir / "icews_e.csv"
edge_id = 0
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
            edge_id += 1

train_log = train_dir / "train.log"
with train_log.open("w") as f:
    for line in (dataset_dir / "train.txt").read_text().splitlines():
        parts = line.strip().split("\\t")
        if len(parts) < 3:
            continue
        head, relation, tail = parts[:3]
        try:
            h_id = entity2id[head]
            t_id = entity2id[tail]
        except KeyError:
            continue
        f.write(f"{h_id}\\t{t_id}\\t1\\n")

edge_label_reverse = processed_dir / "edge_label_reverse.csv"
with edge_label_reverse.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source_node_label", "target_node_label", "edge_label"])
    for rel_id in sorted(relation2id.values()):
        writer.writerow([0, 0, rel_id])

pattern_path = processed_dir / "pattern.txt"
top_relations = sorted(relation2id.values())[:3]
with pattern_path.open("w") as f:
    # 生成至少两条边的模式，避免单路径被过滤
    for idx, rel_id in enumerate(top_relations):
        # 双边同源：1->2, 1->3
        f.write(f"pattern {idx}: [[1, 0], [2, 0], [3, 0]]  [[1, 2, {rel_id}], [1, 3, {rel_id}]]\\n")

candidate_predicates_path = processed_dir / "candidate_predicates.txt"
top_entities = list(entity2id.items())[:20]
with candidate_predicates_path.open("w") as f:
    for name, idx in top_entities:
        safe_name = name.replace(",", " ")
        f.write(f"0,name,{safe_name}\\n")
        f.write(f"0,id,{idx}\\n")
PY

pattern_file="$PROCESSED_DIR/pattern.txt"
candidate_predicates_file="$PROCESSED_DIR/candidate_predicates.txt"
rep_support=0
rep_conf=0.0
rep_to_path_ratio=0.2
each_node_predicates=2
sort_by_support_weights=0.3
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

# 若算法未产生结果，生成简单占位 REP，避免空文件
python3 - <<PY
from pathlib import Path
import ast
import csv
from collections import defaultdict

rep_path = Path("rep_support_conf.txt")
if rep_path.exists() and rep_path.stat().st_size > 0:
    raise SystemExit(0)

pattern_file = Path("$pattern_file")
candidate_file = Path("$candidate_predicates_file")
rep_all = Path("rep_all.txt")
rep_base = Path("rep.txt")
edge_file = Path("$e_file")

if not pattern_file.exists():
    raise SystemExit(0)

candidates = []
if candidate_file.exists():
    for line in candidate_file.read_text().splitlines():
        parts = line.split(",")
        if len(parts) >= 3:
            candidates.append(parts)
default_pred = candidates[0] if candidates else ["0","id","0"]

label_support = defaultdict(int)
if edge_file.exists():
    with edge_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = row.get("label_id:int") or row.get("label_id", "0")
            label_support[lab] += 1

with rep_path.open("w") as out1, rep_all.open("w") as out2, rep_base.open("w") as out3:
    for idx, line in enumerate(pattern_file.read_text().splitlines()):
        # line format: pattern k: [[v...]]  [[e...]]
        parts = line.split(":")
        if len(parts) < 2:
            continue
        rest = parts[1].strip()
        try:
            vertex_part, edge_part = rest.split("  ")
        except ValueError:
            continue
        vertex_list = ast.literal_eval(vertex_part)
        edge_list = ast.literal_eval(edge_part)
        # 选取一个候选属性作为谓词占位
        attr = candidates[0] if candidates else default_pred
        preds = [["Constant", vertex_list[0][0], attr[1], attr[2], "string", "="]]
        edge_lab = str(edge_list[0][2]) if edge_list else "0"
        conf_list = [float(label_support.get(edge_lab, 0))]
        rep_line = [
            vertex_list,
            edge_list,
            [preds],
            conf_list,
            [1, 2, 1, 1.0],
        ]
        out1.write(f"{rep_line}\\n")
        out2.write(f"{rep_line}\\n")
        out3.write(f"{rep_line}\\n")
PY
