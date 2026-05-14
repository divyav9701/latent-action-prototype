"""Export dataset artifacts (JSONL, CSV, NumPy)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def export_jsonl(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_summary_csv(
    summary_rows: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("clip_id,num_frames,num_action_labels,avg_confidence,status\n")
        return
    fieldnames = list(summary_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)


def export_numpy_actions(
    action_ids: Sequence[int] | Any,
    confidences: Sequence[float] | Any,
    path: str | Path,
) -> None:
    """Save a structured array with latent_action and confidence columns."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(action_ids, dtype=np.int32).reshape(-1)
    c = np.asarray(confidences, dtype=np.float64).reshape(-1)
    if a.size == 0 and c.size == 0:
        stacked = np.zeros((0, 2), dtype=np.float64)
    else:
        stacked = np.column_stack([a, c])
    np.save(str(path), stacked)
