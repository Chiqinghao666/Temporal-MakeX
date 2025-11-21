## Temporal-SARL 运行指南（ICEWS14）

本指南展示如何完成 **训练 → 时态路径挖掘 → 本地解释** 全流程，全部基于 ICEWS14 数据集、Temporal-SARL 模块与 pyMakex 图接口。

---

### 1. 训练 TemporalSARL 模型

```bash
cd Makex-main/global_explanations
python train_sarl.py \
  --edge_file ../DataSets/icews14/processed/original_graph/icews_e.csv \
  --dataset_dir ../DataSets/icews14 \
  --num_entities $(jq length ../DataSets/icews14/entity2id.json) \
  --num_relations $(jq length ../DataSets/icews14/relation2id.json) \
  --embed_dim 128 \
  --batch_size 64 \
  --epochs 5 \
  --max_history 5 \
  --min_history 1 \
  --lr 1e-3 \
  --cuda \
  --save_path ./sarl_model.pth
```

说明：
- 训练脚本会自动读取 `train/valid/test.txt`，构建真实历史索引，并进行负采样。
- 训练结束后生成的 `sarl_model.pth` 位于当前目录。

---

### 2. 运行 Temporal-SARL 挖掘（替代 pattern_generator）

```bash
python run_sarl_mining.py \
  --dataset_dir ../DataSets/icews14 \
  --vertex_file ../DataSets/icews14/processed/original_graph/icews_v.csv \
  --edge_file ../DataSets/icews14/processed/original_graph/icews_e.csv \
  --model_path ./sarl_model.pth \
  --num_queries 300 \
  --walks_per_query 60 \
  --max_hops 3 \
  --beam_size 10 \
  --history_size 5 \
  --time_window $((30*86400)) \
  --time_bucket $((7*86400)) \
  --top_signatures 200 \
  --log_dir ./global_explanations \
  --output_rep ./rep_sarl.txt \
  --cuda
```

运行效果：
1. 脚本会采样真实 `(h, r, ?, t)` 查询并调用 `SARLMiner`。终端将输出 `[SARL Step]` / `[Model Decision]` 日志，展示 Transformer 的决策，并在结束时打印 `[SARL Performance]` 统计。
2. 所有原始路径写入 `global_explanations/sarl_raw_paths.txt`（人类可读），聚类后的规则写入 `rep_sarl.txt`，格式与 Makex 原版 REP 完全兼容。

> 可通过 `--time_window` / `--time_bucket` 控制时间约束，`--top_signatures` 控制保留的高频模式数量。

---

### 3. 使用新规则生成本地解释

1. 编辑 `Makex-main/local_explanations/run_local_explanation.sh`，确保：
   ```bash
   rep_file="../global_explanations/rep_sarl.txt"
   ```
2. 执行脚本：
   ```bash
   cd Makex-main/local_explanations
   ./run_local_explanation.sh
   ```
3. Local 模块会基于 SARL 规则输出 `output/icews/` 目录下的 csv/txt，并可配合 `tools/interpret_global.py`、`tools/interpret_local.py` 生成中文报告。

完成上述三步，即可在 ICEWS14 上复现“训练好的 Temporal-SARL → 高质量双星规则 → 本地解释”生产流程。
