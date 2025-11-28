#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-Featured Interpreter for Temporal-SARL Local Explanations.
Features:
1. Reads Scores correctly from CSV.
2. Displays Top-K diversity (Rank 1, Rank 2...).
3. Summarizes pattern structure and semantics.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def default_paths() -> argparse.Namespace:
    # Adjust these relative paths based on your project structure
    repo_root = Path(__file__).resolve().parent.parent
    # Default inputs
    return argparse.Namespace(
        rep_file=repo_root / "global_explanations/rep_sarl.txt",
        vertex_file=repo_root / "DataSets/icews14/processed/original_graph/icews_v.csv",
        relation_file=repo_root / "DataSets/icews14/relation2id.json",
        topk_file=repo_root / "local_explanations/output/icews/topk_rep_id_topk5.csv",
        pairs_file=repo_root / "local_explanations/output/icews/test_sample_pairs.csv",
        subgraph_file=repo_root / "local_explanations/output/icews/subgraph.csv",
        output_file=repo_root / "local_explanations/output/icews/prediction_report.txt",
    )


def load_vertex_names(vertex_path: Path) -> Dict[int, str]:
    """Load mapping from Entity ID to Name (e.g., 3186 -> 'Barack Obama')."""
    id_to_name: Dict[int, str] = {}
    if not vertex_path.exists():
        print(f"[Warning] Vertex file not found: {vertex_path}")
        return id_to_name

    with vertex_path.open("r", encoding="utf-8", errors="replace") as f:
        # Try to handle potential BOM or encoding issues
        reader = csv.DictReader(f)
        for row in reader:
            # Flexible column reading
            vid_str = row.get("vertex_id:int") or row.get("vertex_id")
            name = row.get("name:string") or row.get("name") or ""

            if vid_str is not None:
                try:
                    vid = int(vid_str)
                    id_to_name[vid] = name or f"Entity_{vid}"
                except ValueError:
                    continue
    return id_to_name


def load_relation_map(path: Path) -> Dict[int, str]:
    """Load mapping from Relation ID to Name (e.g., 53 -> 'Consult')."""
    if not path.exists():
        print(f"[Warning] Relation map not found: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Provide readable format
        return {int(idx): name.replace("_", " ") for name, idx in data.items()}
    except Exception as e:
        print(f"[Error] Failed to load relation map: {e}")
        return {}


def load_pairs(pairs_path: Path) -> Dict[int, Tuple[int, int]]:
    """Load test pairs to map pair_id to (user_id, item_id)."""
    pairs: Dict[int, Tuple[int, int]] = {}
    if not pairs_path.exists():
        return pairs
    with pairs_path.open("r") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                u = int(row.get("user_id", -1))
                v = int(row.get("item_id", -1))
                if u != -1:
                    pairs[idx] = (u, v)  # Assuming sequential pair_id matching file row
            except ValueError:
                continue
    return pairs


def load_topk_results(topk_path: Path) -> List[dict]:
    """
    Load the explanation results (CSV).
    Crucial: Reads the 'score' column.
    """
    entries: List[dict] = []
    seen_keys = set()  # Deduplication: (pair_id, rank)

    if not topk_path.exists():
        print(f"[Error] Top-K result file missing: {topk_path}")
        return entries

    print(f"[Info] Loading explanations from: {topk_path}")
    with topk_path.open("r") as f:
        reader = csv.DictReader(f)
        # Debug headers to ensure 'score' exists
        # print(f"[DEBUG] CSV Columns found: {reader.fieldnames}")

        for row in reader:
            try:
                pair_id = int(row["pair_id"])
                # Support 'topk' or 'rank' column name
                rank_val = int(row.get("topk") or row.get("rank", 0))

                # Deduplication check
                unique_key = (pair_id, rank_val)
                if unique_key in seen_keys:
                    continue
                seen_keys.add(unique_key)

                # --- SCORE READING FIX ---
                raw_score = row.get("score") or row.get("explanation_score") or "0.0"
                try:
                    score_float = float(raw_score)
                except ValueError:
                    score_float = 0.0  # Default if parse fails

                # Read Rep ID
                rep_id = int(row.get("rep_id", 0))
                pivot_y = int(row.get("pivot_y", 0))

                entries.append({
                    "pair_id": pair_id,
                    "pivot_y": pivot_y,  # The recommended item
                    "rank": rank_val + 1,  # Convert 0-index to 1-index for display
                    "rep_id": rep_id,
                    "score": score_float
                })
            except (ValueError, KeyError):
                continue

    return entries


def load_patterns(rep_path: Path) -> List[dict]:
    """Load the abstract logic rules from rep_sarl.txt."""
    patterns: List[dict] = []
    if not rep_path.exists():
        return patterns

    with rep_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Safe eval for list format
                entry = ast.literal_eval(line)
                # Structure: [vertices, edges, predicates, stats, meta]
                if len(entry) >= 4:
                    patterns.append({
                        "edges": entry[1],
                        "stats": entry[3],  # [support, confidence]
                        "meta": entry[4] if len(entry) > 4 else []
                    })
            except Exception:
                continue
    return patterns


def describe_pattern_logic(pattern: dict, rel_map: Dict[int, str]) -> str:
    """Convert edge list [[src, dst, rel], ...] to a readable chain string."""
    edges = pattern.get("edges", [])
    if not edges:
        return "Node Attribute Constraints"

    # Heuristic: Try to find a chain
    descriptions = []
    for edge in edges[:4]:  # Limit length for display
        if len(edge) >= 3:
            rel_id = edge[2]
            rel_name = rel_map.get(rel_id, f"Rel_{rel_id}")
            descriptions.append(f"<{rel_id}: {rel_name}>")

    chain = " → ".join(descriptions)
    if len(edges) > 4:
        chain += " ..."
    return chain


def generate_report(
        entries: List[dict],
        pairs: Dict[int, Tuple[int, int]],
        vertex_names: Dict[int, str],
        rel_map: Dict[int, str],
        patterns: List[dict]
) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("      Temporal-SARL Prediction & Explanation Report")
    lines.append("=" * 60)

    # Group by pair_id
    grouped = {}
    for e in entries:
        grouped.setdefault(e["pair_id"], []).append(e)

    # Sort pairs for consistent output
    sorted_pair_ids = sorted(grouped.keys())

    for pair_id in sorted_pair_ids:
        group_entries = grouped[pair_id]
        # Sort by rank
        group_entries.sort(key=lambda x: x["rank"])

        # Get Query Info
        if pair_id in pairs:
            user_id, _ = pairs[pair_id]
            user_name = vertex_names.get(user_id, f"User_{user_id}")
        else:
            user_name = "Unknown_User"

        lines.append(f"\n[Query #{pair_id}] User: {user_name}  (Targeting Top-K items)")
        lines.append("-" * 60)

        for entry in group_entries:
            rank = entry["rank"]
            item_id = entry["pivot_y"]
            item_name = vertex_names.get(item_id, f"Item_{item_id}")
            score = entry["score"]
            rep_id = entry["rep_id"]

            # Format Score
            score_str = f"{score:.4f}" if score > -0.5 else "N/A"

            lines.append(f"  Rank {rank}: Recommended {item_name}")
            lines.append(f"  Confidence Score: {score_str}")

            # Retrieve Pattern Logic
            if 0 <= rep_id < len(patterns):
                pat = patterns[rep_id]
                logic_str = describe_pattern_logic(pat, rel_map)
                support = pat["stats"][0]
                lines.append(f"  [Logic Used (Rule #{rep_id})]: {logic_str}")
                lines.append(f"  [Rule Strength]: Global Support = {int(support)}")
            else:
                lines.append(f"  [Logic]: Rule ID {rep_id} not found in rep file.")

            lines.append("")  # Empty line between ranks

        lines.append("." * 60)

    return "\n".join(lines)


def main():
    args = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk_file", type=Path, help="Path to CSV output", default=args.topk_file)
    parser.add_argument("--output_file", type=Path, help="Path to save report", default=args.output_file)
    parser.add_argument("--rep_file", type=Path, default=args.rep_file)
    parser.add_argument("--vertex_file", type=Path, default=args.vertex_file)
    parser.add_argument("--relation_file", type=Path, default=args.relation_file)
    parser.add_argument("--pairs_file", type=Path, default=args.pairs_file)

    # Parse user args to override defaults
    cli_args = parser.parse_args()

    print("Loading Data...")
    vertex_names = load_vertex_names(cli_args.vertex_file)
    rel_map = load_relation_map(cli_args.relation_file)
    pairs = load_pairs(cli_args.pairs_file)
    patterns = load_patterns(cli_args.rep_file)
    topk_entries = load_topk_results(cli_args.topk_file)

    if not topk_entries:
        print(f"No data found in {cli_args.topk_file}. Please check your local_explanation.py output.")
        return

    print(f"Generating report for {len(topk_entries)} predictions...")
    report_text = generate_report(topk_entries, pairs, vertex_names, rel_map, patterns)

    cli_args.output_file.parent.mkdir(parents=True, exist_ok=True)
    cli_args.output_file.write_text(report_text, encoding="utf-8")

    print(f"Success! Report saved to: {cli_args.output_file}")


if __name__ == "__main__":
    main()