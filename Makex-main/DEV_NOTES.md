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
python pyMakex/test_

temporal_simple.py
```
