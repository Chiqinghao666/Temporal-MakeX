## Temporal-SARL 运行指南

以下步骤演示如何在 ICEWS14 数据集上完成 **训练 → 路径挖掘 → 本地解释** 全流程。

### 1. 训练 TemporalSARL 模型

```bash
cd Makex-main/global_explanations
python train_sarl.py \
  --edge_file ../DataSets/icews14/processed/original_graph/icews_e.csv \
  --num_entities $(jq length ../DataSets/icews14/entity2id.json) \
  --num_relations $(jq length ../DataSets/icews14/relation2id.json) \
  --embed_dim 128 \
  --batch_size 32 \
  --epochs 5 \
  --sample_limit 80000 \
  --save_path ./sarl_model.pth
```

> `num_entities/relations` 可通过 `entity2id.json` / `relation2id.json` 的键数量获得。训练结束会在当前目录生成 `sarl_model.pth`。

### 2. 执行 SARL 挖掘

```bash
python run_sarl_discovery.py \
  --dataset_dir ../DataSets/icews14 \
  --vertex_file ../DataSets/icews14/processed/original_graph/icews_v.csv \
  --edge_file ../DataSets/icews14/processed/original_graph/icews_e.csv \
  --model_path ./sarl_model.pth \
  --output_rep ./rep_sarl.txt \
  --num_queries 300 \
  --walks_per_query 60 \
  --max_hops 3 \
  --beam_size 10 \
  --log_dir ./global_explanations
```

脚本会：

1. 读取 `sarl_model.pth` 和 ICEWS 原始三元组。
2. 采样查询，调用 `SARLMiner` 进行时序路径挖掘，实时输出 `[Progress]` 与 `[SARL Step]`。
3. 依据频率聚类，生成 `rep_sarl.txt`（标准 Makex REP 格式）及 `global_explanations/sarl_raw_paths.txt`（原始路径）。

> 可通过 `--top_signatures` 控制保留的模式数量，`--time_bucket` / `--time_window` 调整时间约束。

### 3. 使用新规则生成本地解释

1. 打开 `local_explanations/run_local_explanation.sh`，将其中的
   ```bash
   rep_file="../global_explanations/rep.txt"
   ```
   修改为
   ```bash
   rep_file="../global_explanations/rep_sarl.txt"
   ```
2. 运行脚本：
   ```bash
   cd Makex-main/local_explanations
   ./run_local_explanation.sh
   ```
3. 生成的解释文件位于 `output/icews/`，可搭配 `tools/interpret_global.py` / `tools/interpret_local.py` 查看中文描述。

按照以上步骤即可完成 Temporal-SARL 的训练、挖掘与解释。***
