# DEV_NOTES

## Stage One – Temporal Graph Backbone

- Added native timestamp storage to `pyMakex`'s `LargeGraph2` edges and exposed helpers in `DataGraphWithInformation` so paths can be filtered by `(ts_min, ts_max)` windows without rebuilding the graph.
- Updated the CSV loader to detect `timestamp/ts/time` columns automatically while reading ICEWS-style edge files and hydrate the new timestamp field during ingestion.
- Extended the Python extension (`pyMakex`) with `GetEdgeTimestamp` and `GetTemporalNeighbors` so higher-level pipelines can query temporal evidence directly from Python.
- Authored a new smoke test (`pyMakex/test_temporal_simple.py`) that builds a toy 5-node, 10-edge temporal graph, verifies timestamp hydration, and exercises the temporal neighbor API.

## Testing

Rebuild the C++ extension and run the new smoke test:

```bash
cd pyMakex && python setup.py build_ext --inplace
python pyMakex/test_temporal_simple.py
```

## 阶段更新 – 可读规则输出（SARL）

- SARL Miner 支持加载 `entity2type.json`（缺失时回退顶点 CSV 的 type 列），规则节点直接写入类型名称，便于展示。
- 新增对 `entity_classification.csv` 的优先支持（字段含 Entity_ID/Type_Name），可直接作为类型来源。
- REP 输出新增查询关系名称、关系链、人类可读类型谓词，以及逐 hop 的时间分箱与时间差谓词；原有关系 ID 仍保留以兼容匹配。
- `run_sarl_mining.py` 与 `run_sarl_discovery.py` 增加 `--entity_type_map`，并将 `time_bucket` 传入 Miner，统一使用 `miner.path_to_rep` 写规则。
- `interpret_global.py` 兼容字符串关系名，生成报告时不再依赖纯数字关系 ID。
