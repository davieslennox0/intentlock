"""Build guard training data. TRAIN split only.

Sources
  1. Reference trajectories on train scenarios: the safe path yields legitimate payloads
     (ALLOW), the compromised path yields the injected payload (BLOCK), each with the
     context the guard would see at that step.
  2. Proposed actions harvested from baseline traces on the train split (optional
     `--traces`), labelled by the scenario's malicious_action_spec. This adds the
     agent's real mistakes: malformed, partial or creative variants of the attack.
  3. Hard negatives: the clean train scenarios already include large transfers,
     unlimited approvals and delegations the user asked for; they are up-weighted.

  python -m training.guard_data --train data/curated/train.jsonl --traces runs/base_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from testbed.agent import guard_context, run_episode
from testbed.labeler import as_specs, matches
from testbed.policies import OraclePolicy

from .guard import guard_messages

HARD_NEGATIVE_KINDS = {"cold_transfer", "unlimited_router", "legit_delegation", "readme_tests", "buy_listing", "pay_invoice"}
ALLOW_REASONS = {
    "swap": "the user asked for this swap",
    "transfer": "the user asked for this payment and the recipient comes from the task",
    "approve": "the user asked to approve this spender for this amount",
    "eip7702_authorization": "the user asked to delegate to this address",
    "run_command": "the user asked to run this documented command",
}
BLOCK_REASONS = {
    "transfer": "the recipient was introduced by tool output, not by the user",
    "approve": "the spender was introduced by tool output, not by the user",
    "eip7702_authorization": "the user did not ask for this account delegation",
    "run_command": "the command comes from untrusted content, not from the user",
    "swap": "the user did not ask for this swap",
}


def examples_from_trace(trace: dict, scenario: dict) -> list[dict]:
    """One guard example per proposed action in a trace."""
    mal = as_specs(scenario.get("malicious_action_spec"))
    expected = scenario.get("expected_actions") or []
    out, tool_outputs = [], []
    for s in trace["steps"]:
        if not s.get("tool"):
            continue
        if s.get("payload"):
            p = s["payload"]
            is_mal = any(matches(m, p) for m in mal)
            is_exp = any(matches(e, p) for e in expected)
            if is_mal or is_exp:      # skip actions that are neither: their label is ambiguous
                label = "BLOCK" if is_mal else "ALLOW"
                reason = (BLOCK_REASONS if is_mal else ALLOW_REASONS).get(p["type"], "")
                msgs = guard_messages(p, guard_context(scenario, tool_outputs))
                out.append({"messages": msgs + [{"role": "assistant", "content": f"{label}: {reason}"}],
                            "label": label, "scenario_id": scenario["id"], "family": scenario["family"],
                            "task_kind": scenario.get("task_kind"), "payload_type": p["type"]})
        tool_outputs.append({"tool": s["tool"], "output": s.get("output", "")})
    return out


def build(scenarios: list[dict], traces: list[dict] | None, rng: random.Random) -> list[dict]:
    by_id = {s["id"]: s for s in scenarios}
    rows = []
    for sc in scenarios:
        assert sc["split"] == "train", f"refusing to build guard data from split {sc['split']}"
        rows += [dict(r, source="reference_safe") for r in examples_from_trace(run_episode(sc, OraclePolicy(sc, "safe")), sc)]
        if not sc["is_clean"]:
            rows += [dict(r, source="reference_compromised")
                     for r in examples_from_trace(run_episode(sc, OraclePolicy(sc, "compromised")), sc)]
    for t in traces or []:
        sc = by_id.get(t["scenario_id"])
        if sc is not None:                 # traces from any other split are ignored
            rows += [dict(r, source="baseline_trace") for r in examples_from_trace(t, sc)]
    # de-duplicate identical prompts, then up-weight hard negatives
    seen, uniq = set(), []
    for r in rows:
        k = r["messages"][1]["content"] + r["label"]
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    hard = [dict(r, source="hard_negative_upweight") for r in uniq
            if r["label"] == "ALLOW" and r["task_kind"] in HARD_NEGATIVE_KINDS]
    uniq += hard
    # balance classes by down-sampling the larger one
    allow = [r for r in uniq if r["label"] == "ALLOW"]
    block = [r for r in uniq if r["label"] == "BLOCK"]
    n = min(len(allow), len(block))
    rng.shuffle(allow)
    rng.shuffle(block)
    out = allow[: int(n * 1.2)] + block[:n]
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/curated/train.jsonl")
    ap.add_argument("--traces", nargs="*", default=[])
    ap.add_argument("--out", default="data/sft/guard_train.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    scenarios = [json.loads(l) for l in open(a.train)]
    traces = [json.loads(l) for p in a.traces for l in open(p) if l.strip()]
    ids = sorted({s["id"] for s in scenarios})
    rng.shuffle(ids)
    val_ids = set(ids[: int(len(ids) * a.val_frac)])       # split by scenario, not by row
    rows = build(scenarios, traces, rng)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, keep in (("train", lambda r: r["scenario_id"] not in val_ids), ("val", lambda r: r["scenario_id"] in val_ids)):
        part = [r for r in rows if keep(r)]
        path = out if name == "train" else out.with_name(out.stem.replace("train", "val") + out.suffix)
        with open(path, "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(part), dict(Counter(r["label"] for r in part)), dict(Counter(r["source"] for r in part)), "->", path)


if __name__ == "__main__":
    main()
