#!/bin/bash
rm output/icews/local_topk1.txt
set -euo pipefail

# 适配 ICEWS：目录与路径
mkdir -p output/icews reserve_rep/icews

# 生成简单的测试样本（user_id,item_id）与子图文件，来自 ICEWS 边表前 100 条
python3 - <<'PY'
from pathlib import Path
import csv

root = Path(__file__).resolve().parent
edge_path = (root / "../DataSets/icews14/processed/original_graph/icews_e.csv").resolve()
output_dir = (root / "output/icews")
output_dir.mkdir(parents=True, exist_ok=True)

test_pairs = output_dir / "test_sample_pairs.csv"
subgraph = output_dir / "subgraph.csv"
subgraph_pivot = output_dir / "subgraph_pivot.csv"
edge_label_reverse = output_dir / "edge_label_reverse.csv"

# 创建边反向文件（简单占位，用 src_type/dst_type=0）
with edge_label_reverse.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["src_type","reverse_type"])
    writer.writerow([0,0])

pairs = []
with edge_path.open() as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 100:
            break
        pairs.append((int(row["source_id:int"]), int(row["target_id:int"]), int(row["label_id:int"])))

# 测试对文件
with test_pairs.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id","item_id"])
    for u,v,_ in pairs:
        writer.writerow([u,v])

# 子图文件（简单用一条边作为子图）
with subgraph.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["graph_id","src_id","dst_id","src_type","dst_type","edge_type"])
    for gid,(u,v,lab) in enumerate(pairs):
        writer.writerow([gid,u,v,0,0,lab])

with subgraph_pivot.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["graph_id","pivot_x","pivot_y"])
    for gid,(u,v,_) in enumerate(pairs):
        writer.writerow([gid,u,v])
PY

pattern_nums=(50)
conf_limits=(0.7)
supp_limits=(150000)
each_pattern_rep_nums=(5)
random_seeds=(1996)
rep_num_ratios=(1.0)
vary_topk=(1 5 10 15)

rep_file="../global_explanations/rep.txt"
output_dir="./output/icews/"
reserve_rep_dir="./reserve_rep/icews/"
test_pairs_file="./output/icews/test_sample_pairs.csv"
subgraph_path="./output/icews/subgraph.csv"
subgraph_pivot_path="./output/icews/subgraph_pivot.csv"
edge_label_reverse="./output/icews/edge_label_reverse.csv"
v_path="../DataSets/icews14/processed/original_graph/icews_v.csv"
e_path="../DataSets/icews14/processed/original_graph/icews_e.csv"

for pattern_num in "${pattern_nums[@]}"; do
  for conf_limit in "${conf_limits[@]}"; do
    for supp_limit in "${supp_limits[@]}"; do
      for each_pattern_rep_num in "${each_pattern_rep_nums[@]}"; do
        for rep_num_ratio in "${rep_num_ratios[@]}"; do
          for random_seed in "${random_seeds[@]}"; do
            for topk in "${vary_topk[@]}"; do
              makex_explanation_v="${output_dir}v_topk${topk}.csv"
              makex_explanation_e="${output_dir}e_topk${topk}.csv"
              topk_rep_id_file="${output_dir}topk_rep_id_topk${topk}.csv"
              reserve_rep_file="${reserve_rep_dir}rep_topk${topk}.txt"
              output_file="${output_dir}local_topk${topk}.csv"
              output_file_txt="${output_dir}local_topk${topk}.txt"

              echo "Parameters passed to script: "
              echo "topk: $topk"
              echo "rep_file: $rep_file"
              echo "pattern_num: $pattern_num"
              echo "conf_limit: $conf_limit"
              echo "supp_limit: $supp_limit"
              echo "each_pattern_rep_num: $each_pattern_rep_num"
              echo "rep_num_ratio: $rep_num_ratio"
              echo "topk_rep_id_file: $topk_rep_id_file"
              echo "makex_explanation_v: $makex_explanation_v"
              echo "makex_explanation_e: $makex_explanation_e"
              echo "random_seed: $random_seed"
              echo "output_file: $output_file"
              echo "reserve_rep_file: $reserve_rep_file"
              echo "output_file_txt: $output_file_txt"

              ./local_explanation.sh "$pattern_num" "$conf_limit" "$supp_limit" "$each_pattern_rep_num" "$rep_file" "$rep_num_ratio" "$topk_rep_id_file" "$makex_explanation_v" "$makex_explanation_e" "$topk" "$random_seed" "$output_file" "$reserve_rep_file" "$test_pairs_file" "$subgraph_path" "$subgraph_pivot_path" "$edge_label_reverse" "$v_path" "$e_path" >> "$output_file_txt" 2>&1
            done
          done
        done
      done
    done
  done
done
