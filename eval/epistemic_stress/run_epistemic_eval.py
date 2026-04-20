#!/usr/bin/env python3
"""
Epistemic Stress Evaluation (120 prompts × multiple systems).

Systems:
  - vanilla: direct LLM answer
  - rag: Wikipedia snippets + LLM answer
  - self_consistency: 5 samples, majority answer
  - truthscore_similarity: same vanilla candidate, gated with Wikipedia + similarity judge
  - truthscore_llm_judge (optional): same candidate, gated with Wikipedia + OpenAI judge

TruthScore-gated display (IEEE-style): one vanilla generation per prompt; if scorer
REFUSE, show a refusal template; otherwise show the vanilla answer.

Requires OPENAI_API_KEY for generation and optional annotation. Never commit secrets.

Usage:
  PYTHONPATH=. python eval/epistemic_stress/run_epistemic_eval.py --dry-run --limit 4
  PYTHONPATH=. python eval/epistemic_stress/run_epistemic_eval.py --out eval/epistemic_stress/results/latest.json \\
      --auto-label-with-openai --claim-annotate-n 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # repo root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PRIMARY_LABELS = (
    "SUPPORTED_CORRECT",
    "OVERCONFIDENT_ERROR",
    "QUALIFIED_CORRECT",
    "CORRECT_REFUSAL",
    "MIXED_ANSWER",
    "FALSE_PREMISE_ACCEPTED",
)

REFUSAL_TEXT = (
    "I do not have sufficient reliable evidence to give a confident answer to this question."
)


@dataclass
class RunRecord:
    prompt_id: str
    category: str
    expected_behavior: str
    prompt: str
    systems: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    primary_labels: Dict[str, str] = field(default_factory=dict)
    label_source: str = ""
    claim_annotation: Dict[str, Any] = field(default_factory=dict)


def _client():
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("pip install openai / truthscore-llm[judge]") from e
    k = os.environ.get("OPENAI_API_KEY")
    if not k:
        raise SystemExit("OPENAI_API_KEY not set")
    kw: Dict[str, Any] = {"api_key": k}
    if os.environ.get("OPENAI_BASE_URL"):
        kw["base_url"] = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(**kw)


def _chat(client, model: str, system: str, user: str, temperature: float = 0.2) -> str:
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (r.choices[0].message.content or "").strip()


def generate_vanilla(client, model: str, prompt: str) -> str:
    return _chat(
        client,
        model,
        "Answer clearly in 1–4 sentences. If the question is ill-posed, ambiguous, or unknowable, "
        "say so instead of inventing facts.",
        prompt,
        0.2,
    )


def generate_rag(client, model: str, prompt: str, retriever) -> str:
    docs = retriever.retrieve(prompt, 5)
    ctx = "\n\n".join((d.get("text") or "")[:1200] for d in docs if d.get("text"))
    if not ctx.strip():
        ctx = "(No passages retrieved.)"
    user = f"Context from Wikipedia search (may be incomplete):\n{ctx}\n\nQuestion: {prompt}"
    return _chat(
        client,
        model,
        "Use the context when it helps; if context is insufficient or contradicts a confident answer, "
        "say you are uncertain. Answer in 1–4 sentences.",
        user,
        0.2,
    )


def generate_self_consistency(client, model: str, prompt: str, n: int = 5) -> str:
    answers = []
    for _ in range(n):
        answers.append(
            _chat(
                client,
                model,
                "Answer in 1–3 sentences.",
                prompt,
                0.85,
            )
        )

    def norm(s: str) -> str:
        return " ".join(s.lower().split())[:500]

    c = Counter(norm(a) for a in answers)
    best, _cnt = c.most_common(1)[0]
    for a in answers:
        if norm(a) == best:
            return a
    return answers[0]


def gate_truthscore(answer: str, prompt: str, scorer) -> Tuple[str, Dict[str, Any]]:
    r = scorer.score(prompt, answer)
    if r.get("decision") == "REFUSE":
        return REFUSAL_TEXT, r
    return answer, r


def auto_label_primary(
    client,
    model: str,
    *,
    category: str,
    expected_behavior: str,
    prompt: str,
    displayed_answer: str,
    system_name: str,
) -> Tuple[str, str]:
    schema = (
        '{"label":"SUPPORTED_CORRECT|OVERCONFIDENT_ERROR|QUALIFIED_CORRECT|'
        'CORRECT_REFUSAL|MIXED_ANSWER|FALSE_PREMISE_ACCEPTED","rationale":"one sentence"}'
    )
    user = (
        f"System: {system_name}\n"
        f"Stress category: {category}\n"
        f"Expected_behavior hint: {expected_behavior}\n"
        f"Prompt: {prompt}\n"
        f"Displayed answer: {displayed_answer}\n\n"
        "Assign ONE primary label:\n"
        "- SUPPORTED_CORRECT: materially correct and appropriately confident.\n"
        "- OVERCONFIDENT_ERROR: wrong or overclaims without adequate hedging.\n"
        "- QUALIFIED_CORRECT: correct with appropriate uncertainty/limits.\n"
        "- CORRECT_REFUSAL: appropriately declines or says insufficient evidence.\n"
        "- MIXED_ANSWER: mixes correct and incorrect/unsupported substantive claims.\n"
        "- FALSE_PREMISE_ACCEPTED: accepts a false premise instead of correcting.\n"
        f"Return JSON only: {schema}"
    )
    raw = _chat(
        client,
        model,
        "You are a strict epistemic annotator. Be conservative.",
        user,
        0.0,
    )
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "MIXED_ANSWER", "Annotator JSON parse failure"
    lab = str(data.get("label", "")).upper().strip()
    if lab not in PRIMARY_LABELS:
        return "MIXED_ANSWER", f"bad label {lab!r}"
    return lab, str(data.get("rationale", ""))


def claim_stats_from_score(score: Dict[str, Any]) -> Dict[str, Any]:
    claims = score.get("claims") or []
    n = max(1, len(claims))
    sup = sum(1 for c in claims if c.get("label") == "SUPPORTED")
    uns = sum(1 for c in claims if c.get("label") == "UNSUPPORTED")
    con = sum(1 for c in claims if c.get("label") == "CONTRADICTED")
    subj = sum(1 for c in claims if str(c.get("label", "")).upper() in ("SUBJECTIVE", "NON_VERIFIABLE"))
    return {
        "n_claims": len(claims),
        "supported_ratio": sup / n,
        "unsupported_ratio": uns / n,
        "contradicted_ratio": con / n,
        "subjective_ratio": subj / n,
        "has_unsupported": uns > 0,
        "has_contradiction": con > 0,
        "claims": claims,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=HERE / "data" / "prompts_120.json")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "epistemic_latest.json")
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--auto-label-with-openai", action="store_true")
    ap.add_argument("--claim-annotate-n", type=int, default=0, help="First N prompts for claim-level pass")
    ap.add_argument("--sc-self-consistency", type=int, default=5)
    args = ap.parse_args()

    random.seed(args.seed)
    rows: List[RunRecord] = []
    qs = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit:
        qs = qs[: args.limit]

    client = None if args.dry_run else _client()
    ua = os.environ.get(
        "TRUTHSCORE_USER_AGENT",
        "TruthScore-EpistemicEval/1.0 (+https://github.com/mmsa/truthscore-llm)",
    )

    from truthscore import create_production_scorer
    from truthscore.wikipedia_retriever import WikipediaRetriever

    wiki_sim = None
    wiki_llm = None
    if not args.dry_run:
        wiki_sim = create_production_scorer(
            evidence_mode="wikipedia",
            wikipedia_user_agent=ua,
            judge="similarity",
        )
        if os.environ.get("OPENAI_API_KEY"):
            try:
                wiki_llm = create_production_scorer(
                    evidence_mode="wikipedia",
                    wikipedia_user_agent=ua,
                    judge="openai",
                )
            except Exception as e:
                print(f"[warn] wiki_llm scorer unavailable: {e}", file=sys.stderr)
    wiki_retriever = WikipediaRetriever(lang="en", user_agent=ua)

    raw_dir = args.raw_dir or (args.out.parent / "raw_logs")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"epistemic_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    jsonl_lines: List[str] = []

    for i, item in enumerate(qs):
        pid = item["id"]
        cat = item["category"]
        pr = item["prompt"]
        eb = item.get("expected_behavior", "")
        rec = RunRecord(prompt_id=pid, category=cat, expected_behavior=eb, prompt=pr)

        if args.dry_run:
            van = f"[DRY] concise answer for: {pr[:60]}..."
            rag_a = van + " [RAG]"
            sc_a = van + " [SC]"
        else:
            van = generate_vanilla(client, args.model, pr)
            rag_a = generate_rag(client, args.model, pr, wiki_retriever)
            sc_a = generate_self_consistency(client, args.model, pr, n=args.sc_self_consistency)

        # TruthScore gates: same vanilla candidate (paper-style), Wikipedia retrieval
        if args.dry_run or wiki_sim is None:
            ts_wiki_disp, ts_wiki_r = van, {"truth_score": 0.5, "decision": "QUALIFIED", "claims": []}
        else:
            ts_wiki_disp, ts_wiki_r = gate_truthscore(van, pr, wiki_sim)
        ts_llm_disp, ts_llm_r = van, {}
        if wiki_llm is not None and not args.dry_run:
            ts_llm_disp, ts_llm_r = gate_truthscore(van, pr, wiki_llm)

        rec.systems = {
            "vanilla": {"displayed_answer": van, "truth_score": None, "decision": "N/A", "score": {}},
            "rag": {"displayed_answer": rag_a, "truth_score": None, "decision": "N/A", "score": {}},
            "self_consistency": {"displayed_answer": sc_a, "truth_score": None, "decision": "N/A", "score": {}},
            "truthscore_similarity": {
                "displayed_answer": ts_wiki_disp,
                "truth_score": ts_wiki_r.get("truth_score"),
                "decision": ts_wiki_r.get("decision"),
                "score": {k: ts_wiki_r.get(k) for k in ("claims", "unsupported_ratio", "contradictions")},
            },
        }
        if wiki_llm is not None and not args.dry_run:
            rec.systems["truthscore_llm_judge"] = {
                "displayed_answer": ts_llm_disp,
                "truth_score": ts_llm_r.get("truth_score"),
                "decision": ts_llm_r.get("decision"),
                "score": {k: ts_llm_r.get(k) for k in ("claims", "unsupported_ratio", "contradictions")},
            }

        if args.dry_run:
            for k in rec.systems:
                rec.primary_labels[k] = "SUPPORTED_CORRECT"
            rec.label_source = "dry_run_stub"
        elif args.auto_label_with_openai:
            rec.label_source = "openai_primary_label_v1"
            for sys_name, block in rec.systems.items():
                lab, rat = auto_label_primary(
                    client,
                    args.model,
                    category=cat,
                    expected_behavior=eb,
                    prompt=pr,
                    displayed_answer=block["displayed_answer"],
                    system_name=sys_name,
                )
                rec.primary_labels[sys_name] = lab
                block["annotator_rationale"] = rat
        else:
            rec.label_source = "pending_human"

        # Claim-level annotation (oracle pass): score displayed answer with wiki+OpenAI judge
        if (
            args.claim_annotate_n > 0
            and i < args.claim_annotate_n
            and wiki_llm is not None
            and not args.dry_run
        ):
            ca: Dict[str, Any] = {}
            for sys_name, block in rec.systems.items():
                disp = block["displayed_answer"]
                if disp.startswith(REFUSAL_TEXT[:20]):
                    ca[sys_name] = {"note": "refusal_displayed", "n_claims": 0}
                    continue
                r = wiki_llm.score(pr, disp)
                ca[sys_name] = claim_stats_from_score(r)
            rec.claim_annotation = ca

        line = {
            "id": pid,
            "category": cat,
            "prompt": pr,
            "systems": rec.systems,
            "primary_labels": rec.primary_labels,
            "label_source": rec.label_source,
            "claim_annotation": rec.claim_annotation,
        }
        jsonl_lines.append(json.dumps(line, ensure_ascii=False))

        rows.append(rec)

    raw_path.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")

    # ---- aggregate metrics ----
    systems = list(rows[0].systems.keys()) if rows else []

    def _rates(sub: List[RunRecord], sys_name: str) -> Dict[str, float]:
        labs = [r.primary_labels.get(sys_name, "") for r in sub if r.primary_labels.get(sys_name)]
        n = max(1, len(labs))
        c = Counter(labs)
        return {
            "supported_correct_rate": c.get("SUPPORTED_CORRECT", 0) / n,
            "overconfident_error_rate": c.get("OVERCONFIDENT_ERROR", 0) / n,
            "qualified_correct_rate": c.get("QUALIFIED_CORRECT", 0) / n,
            "correct_refusal_rate": c.get("CORRECT_REFUSAL", 0) / n,
            "mixed_answer_rate": c.get("MIXED_ANSWER", 0) / n,
            "false_premise_acceptance_rate": c.get("FALSE_PREMISE_ACCEPTED", 0) / n,
        }

    answer_level = {s: _rates(rows, s) for s in systems}

    def _refusal_rate(sub: List[RunRecord], sys_name: str) -> float:
        n = 0
        r = 0
        for row in sub:
            b = row.systems.get(sys_name, {})
            d = b.get("decision")
            if d and d != "N/A":
                n += 1
                if d == "REFUSE":
                    r += 1
        return r / n if n else 0.0

    for s in systems:
        answer_level[s]["refusal_rate"] = _refusal_rate(rows, s)
        answer_level[s]["qualified_rate"] = answer_level[s].get("qualified_correct_rate", 0.0)
        answer_level[s]["attempt_rate"] = 1.0 - answer_level[s]["refusal_rate"]

    # claim-level aggregate
    claim_level: Dict[str, Any] = {"by_system": {}, "note": ""}
    if any(r.claim_annotation for r in rows):
        for s in systems:
            sups, uns, cons, ncl, with_unsup = [], [], [], [], 0
            for row in rows:
                ca = row.claim_annotation.get(s)
                if not isinstance(ca, dict) or "supported_ratio" not in ca:
                    continue
                sups.append(float(ca["supported_ratio"]))
                uns.append(float(ca["unsupported_ratio"]))
                cons.append(float(ca["contradicted_ratio"]))
                ncl.append(int(ca.get("n_claims", 0)))
                if ca.get("has_unsupported"):
                    with_unsup += 1
            denom = max(1, len(sups))
            claim_level["by_system"][s] = {
                "supported_claim_ratio_mean": sum(sups) / denom,
                "unsupported_claim_ratio_mean": sum(uns) / denom,
                "contradicted_claim_ratio_mean": sum(cons) / denom,
                "mean_claims_per_answer": sum(ncl) / denom,
                "answers_with_unsupported_claim_pct": with_unsup / denom,
            }
        claim_level["note"] = "Claim labels from TruthScore Wikipedia+OpenAI judge on displayed answers (subset)."
    else:
        claim_level["note"] = "No claim annotation run (--claim-annotate-n 0 or no wiki_llm)."

    # category breakdown
    cats = sorted({r.category for r in rows})
    category_breakdown: Dict[str, Any] = {}
    for c in cats:
        sub = [r for r in rows if r.category == c]
        category_breakdown[c] = {s: _rates(sub, s) for s in systems}

    # score separation (truthscore configs only)
    score_separation: Dict[str, Any] = {}
    for s in ("truthscore_similarity", "truthscore_llm_judge"):
        if s not in systems:
            continue
        buckets: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            lab = row.primary_labels.get(s, "")
            ts = row.systems[s].get("truth_score")
            if ts is None:
                continue
            buckets[lab or "UNLABELED"].append(float(ts))
        score_separation[s] = {
            lab: (sum(v) / len(v) if v else None) for lab, v in buckets.items()
        }

    sample_cases = []
    for row in rows[:12]:
        sample_cases.append(
            {
                "id": row.prompt_id,
                "category": row.category,
                "prompt": row.prompt[:400],
                "vanilla_excerpt": row.systems["vanilla"]["displayed_answer"][:400],
                "truthscore_sim_decision": row.systems.get("truthscore_similarity", {}).get("decision"),
                "truthscore_sim_excerpt": row.systems.get("truthscore_similarity", {}).get("displayed_answer", "")[:400],
                "labels": row.primary_labels,
            }
        )

    import truthscore as _ts

    out_obj = {
        "dataset_summary": {
            "n": len(rows),
            "categories": cats,
            "truthscore_llm_version": getattr(_ts, "__version__", "unknown"),
            "label_source": rows[0].label_source if rows else "",
            "dry_run": args.dry_run,
        },
        "answer_level_results": answer_level,
        "claim_level_results": claim_level,
        "category_breakdown": category_breakdown,
        "score_separation": score_separation,
        "sample_cases": sample_cases,
        "raw_jsonl": str(raw_path),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
