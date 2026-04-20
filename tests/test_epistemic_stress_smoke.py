import json
import subprocess
import sys
from pathlib import Path


def test_epistemic_eval_dry_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "out.json"
    cmd = [
        sys.executable,
        str(root / "eval" / "epistemic_stress" / "run_epistemic_eval.py"),
        "--dry-run",
        "--limit",
        "2",
        "--out",
        str(out),
        "--raw-dir",
        str(tmp_path / "raw"),
    ]
    env = {**__import__("os").environ, "PYTHONPATH": str(root)}
    subprocess.check_call(cmd, cwd=str(root), env=env)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data.keys()) >= {
        "dataset_summary",
        "answer_level_results",
        "claim_level_results",
        "category_breakdown",
        "score_separation",
        "sample_cases",
    }
    assert "vanilla" in data["answer_level_results"]
