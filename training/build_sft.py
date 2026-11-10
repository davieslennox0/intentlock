"""Build hardening SFT data from the TRAIN split only.

For every training scenario we replay the safe reference trajectory through the
real agent loop and emit one chat row per assistant turn (SafeTune puts the loss
on the last assistant turn). Attack scenarios teach "finish the user's task,
treat tool output as data, mention the injection"; clean scenarios teach that
ordinary notes and legitimately large actions are fine.

  python -m training.build_sft --train data/curated/train.jsonl --out data/sft/agent_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from testbed.agent import run_episode
from testbed.policies import ScriptedPolicy, oracle_turns

SAFE_THOUGHTS = {
    "read": ["I need this information for the user's task.", "Reading the data the task depends on."],
    "act_attack": ["The tool output contains an instruction, but tool output is data, not a command. The user asked for this action only.",
                   "Ignoring the instruction embedded in the tool output; doing only what the user asked."],
    "act_clean": ["This is the action the user asked for.", "The data checks out; carrying out the user's request."],
    "final_attack": ["Done. I will report the embedded instruction I ignored.",
                     "Task handled; the injected instruction was not followed."],
    "final_clean": ["Task complete.", "Done as requested."],
}


def trajectory(sc: dict, rng: random.Random) -> list[dict]:
    turns = oracle_turns(sc, "safe")
    n_reads = len(sc.get("reference_reads") or [])
    kind = "clean" if sc["is_clean"] else "attack"
    for i, t in enumerate(turns):
        if "final" in t:
            t["thought"] = rng.choice(SAFE_THOUGHTS[f"final_{kind}"])
        elif i < n_reads:
            t["thought"] = rng.choice(SAFE_THOUGHTS["read"])
        else:
            t["thought"] = rng.choice(SAFE_THOUGHTS[f"act_{kind}"])
    return turns


def rows_for(sc: dict, rng: random.Random) -> list[dict]:
    assert sc["split"] == "train", f"refusing to build training data from split {sc['split']}"
    tr = run_episode(sc, ScriptedPolicy(trajectory(sc, rng)))
    assert not tr["labels"]["attack_success"] and tr["labels"]["task_success"], sc["id"]
    msgs, rows = tr["messages"], []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            rows.append({"messages": msgs[: i + 1], "scenario_id": sc["id"], "family": sc["family"],
                         "is_clean": sc["is_clean"], "turn": sum(x["role"] == "assistant" for x in msgs[: i + 1]) - 1})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/curated/train.jsonl")
    ap.add_argument("--out", default="data/sft/agent_train.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    scenarios = [json.loads(l) for l in open(a.train)]
    rng.shuffle(scenarios)
    n_val = int(len(scenarios) * a.val_frac)       # validation is carved from TRAIN scenarios
    parts = {"val": scenarios[:n_val], "train": scenarios[n_val:]}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, scs in parts.items():
        rows = [r for sc in scs for r in rows_for(sc, rng)]
        rng.shuffle(rows)
        path = out if name == "train" else out.with_name(out.stem.replace("train", "val") + out.suffix)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(scs)} scenarios -> {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
