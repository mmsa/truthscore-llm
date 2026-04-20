"""Smoke test for eval harness (no API)."""

import json
import subprocess
import sys
from pathlib import Path


def test_eval_dry_run_produces_valid_json(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "out.json"
    cmd = [
        sys.executable,
        str(root / "eval" / "run_claim_evaluation.py"),
        "--dry-run",
        "--limit",
        "2",
        "--out",
        str(out),
        "--raw-dir",
        str(tmp_path / "raw"),
    ]
    env = {**dict(**__import__("os").environ), "PYTHONPATH": str(root)}
    subprocess.check_call(cmd, cwd=str(root), env=env)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data.keys()) >= {
        "summary",
        "confusion_matrix",
        "score_distribution",
        "category_breakdown",
        "sample_outputs",
    }
    assert "config_A" in data["summary"]
