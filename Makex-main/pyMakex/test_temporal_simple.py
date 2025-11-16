#!/usr/bin/env python3
"""Temporal graph smoke test for Makex C++ backend."""

import os
import tempfile
import unittest

import pyMakex


class TemporalGraphSmokeTest(unittest.TestCase):
    """Constructs a tiny temporal graph and validates timestamp APIs."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.vertex_file = os.path.join(self._tmp_dir.name, "vertices.csv")
        self.edge_file = os.path.join(self._tmp_dir.name, "edges.csv")
        self._write_vertices()
        self._write_edges()
        self.graph_ptr = pyMakex.ReadDataGraph(self.vertex_file, self.edge_file)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _write_vertices(self):
        vertices = [
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
        ]
        with open(self.vertex_file, "w", encoding="utf-8") as fout:
            fout.write("vertex_id:int,label_id:int\n")
            for vertex_id, label_id in vertices:
                fout.write(f"{vertex_id},{label_id}\n")

    def _write_edges(self):
        # (edge_id, src, dst, label, timestamp)
        self.edges = [
            (0, 0, 1, 0, 100),
            (1, 0, 2, 0, 200),
            (2, 0, 3, 1, 500),
            (3, 1, 0, 1, 120),
            (4, 1, 4, 1, 260),
            (5, 2, 4, 0, 320),
            (6, 2, 3, 0, 360),
            (7, 3, 4, 1, 600),
            (8, 4, 0, 0, 710),
            (9, 4, 2, 1, 900),
        ]
        with open(self.edge_file, "w", encoding="utf-8") as fout:
            fout.write(
                "edge_id:int,source_id:int,target_id:int,label_id:int,timestamp:int\n"
            )
            for row in self.edges:
                fout.write(
                    f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n"
                )
        self.edge_ts_map = {edge_id: ts for edge_id, *_unused, ts in self.edges}

    def test_edge_count_and_timestamps(self):
        edges = pyMakex.GetAllEdge(self.graph_ptr)
        self.assertEqual(len(edges), len(self.edges))
        for edge_id, expected_ts in self.edge_ts_map.items():
            observed_ts = pyMakex.GetEdgeTimestamp(self.graph_ptr, edge_id)
            self.assertEqual(
                observed_ts,
                expected_ts,
                msg=f"edge {edge_id} timestamp mismatch",
            )

    def test_temporal_neighbor_queries(self):
        neighbors = pyMakex.GetTemporalNeighbors(self.graph_ptr, 0, 50, 250, 1)
        self.assertSetEqual(set(neighbors), {1, 2})

        neighbors = pyMakex.GetTemporalNeighbors(self.graph_ptr, 0, 260, 800, 1)
        self.assertSetEqual(set(neighbors), {3})

        # Exclusive range should drop boundary timestamps
        exclusive_neighbors = pyMakex.GetTemporalNeighbors(
            self.graph_ptr, 0, 100, 500, 0
        )
        self.assertSetEqual(set(exclusive_neighbors), {2})

        empty_neighbors = pyMakex.GetTemporalNeighbors(
            self.graph_ptr, 3, 0, 50, 1
        )
        self.assertEqual(empty_neighbors, [])


if __name__ == "__main__":
    unittest.main()
