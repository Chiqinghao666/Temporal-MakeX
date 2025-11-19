#!/bin/bash

pattern_num=$1
conf_limit=$2
supp_limit=$3
each_pattern_rep_num=$4
rep_file=$5
rep_num_ratio=$6
topk_rep_id_file=$7
makex_explanation_v=$8
makex_explanation_e=$9
topk=${10}
reserve_rep_file=${11}
output_file_txt=${12}
test_pairs_file=${13}
subgraph_path=${14}
subgraph_pivot_path=${15}
edge_label_reverse=${16}
v_path=${17}
e_path=${18}

echo "pattern_num: $pattern_num"
echo "conf_limit: $conf_limit"
echo "supp_limit: $supp_limit"
echo "each_pattern_rep_num: $each_pattern_rep_num"
echo "rep_file: $rep_file"
echo "rep_num_ratio: $rep_num_ratio"
echo "topk_rep_id_file: $topk_rep_id_file"
echo "makex_explanation_v: $makex_explanation_v"
echo "makex_explanation_e: $makex_explanation_e"
echo "reserve_rep_file: $reserve_rep_file"
echo "output_file_txt: $output_file_txt"
echo "test_pairs_file: $test_pairs_file"
echo "subgraph_path: $subgraph_path"
echo "subgraph_pivot_path: $subgraph_pivot_path"
echo "edge_label_reverse: $edge_label_reverse"
echo "v_path: $v_path"
echo "e_path: $e_path"

export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

python ./local_explanation.py \
--pattern_num "$pattern_num" \
--conf_limit "$conf_limit" \
--supp_limit "$supp_limit" \
--each_pattern_rep "$each_pattern_rep_num" \
--rep_file "$rep_file" \
--rep_num_ratio "$rep_num_ratio" \
--topk_rep_id_file "$topk_rep_id_file" \
--makex_explanation_v "$makex_explanation_v" \
--makex_explanation_e "$makex_explanation_e" \
--topk "$topk" \
--reserve_rep_file "$reserve_rep_file" \
--edge_label_reverse_csv "$edge_label_reverse" \
--v_path "$v_path" \
--e_path "$e_path" \
--subgraph_path "$subgraph_path" \
--subgraph_pivot_path "$subgraph_pivot_path" \
--ml_path ../DataSets/icews14/processed/train_test/train.log \
--delta_l 0.0  --delta_r 1.0 \
--sort_criteria conf \
--has_ml 1 \
--max_degree 3 --max_length 2 --max_subgraph_to_pattern_num 2 \
--hop_decay_factor 0.8 \
--enable_topk 1 \
--sample_pair_num 1000 \
--test_sample_pairs_file "$test_pairs_file" \
--pattern_file pattern.txt
