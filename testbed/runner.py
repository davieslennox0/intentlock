"""Run scenarios through the agent and write traces as JSONL.

  python -m testbed.runner --scenarios data/curated/test.jsonl --out runs/base_test.jsonl
  python -m testbed.runner --scenarios data/seed/handwritten.jsonl --policy oracle-compromised --out runs/x.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from .agent import run_episode
from .policies import DEFAULT_MODEL, OraclePolicy


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_many(scenarios, policy_factory, *, out=None, config_name="base", wrapper=None,
             boundary_note=False, guard=None, seed=0, progress=True, resume=True):
    done = {}
    if out and resume and os.path.exists(out):
        done = {r["scenario_id"]: r for r in load_jsonl(out)}
    Path(out).parent.mkdir(parents=True, exist_ok=True) if out else None
    fh = open(out, "a") if out else None
    traces = list(done.values())
    for i, sc in enumerate(scenarios):
        if sc["id"] in done:
            continue
        tr = run_episode(sc, policy_factory(sc), wrapper=wrapper, boundary_note=boundary_note,
                         guard=guard, config_name=config_name, seed=seed)
        traces.append(tr)
        if fh:
            fh.write(json.dumps(tr, ensure_ascii=False) + "\n")
            fh.flush()
        if progress and (i + 1) % 20 == 0:
            n_att = sum(1 for t in traces if not t["is_clean"])
            asr = sum(t["labels"]["attack_success"] for t in traces if not t["is_clean"]) / max(1, n_att)
            print(f"[{config_name}] {i + 1}/{len(scenarios)}  running ASR={asr:.2%}", flush=True)
    if fh:
        fh.close()
    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="hf", choices=["hf", "oracle-safe", "oracle-compromised"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--guard-adapter", default=None, help="LoRA guard adapter; enables the pre-signing guard")
    ap.add_argument("--config-name", default="base")
    ap.add_argument("--wrapper", default=None, choices=[None, "default", "alt", "none"],
                    help="override the scenario's own wrapper")
    ap.add_argument("--boundary-note", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: run only every n-th scenario starting at i (for parallel CI jobs)")
    a = ap.parse_args()

    scenarios = load_jsonl(a.scenarios)[: a.limit]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        scenarios = scenarios[i::n]
    guard = None
    hub = None
    if a.policy == "hf":
        from .policies import HFPolicy, ModelHub
        hub = ModelHub(a.model, load_in_4bit=a.load_in_4bit)
        if a.adapter:
            hub.add_adapter("agent", a.adapter)
        pol = HFPolicy(hub, adapter="agent" if a.adapter else None, temperature=a.temperature, seed=a.seed)
        factory = lambda sc: pol
        if a.guard_adapter:
            from training.guard import GuardModel
            hub.add_adapter("guard", a.guard_adapter)
            guard = GuardModel(hub, adapter="guard")
    else:
        mode = a.policy.split("-")[1]
        factory = lambda sc: OraclePolicy(sc, mode)

    t0 = time.time()
    traces = run_many(scenarios, factory, out=a.out, config_name=a.config_name, wrapper=a.wrapper,
                      boundary_note=a.boundary_note, guard=guard, seed=a.seed)
    meta = {"config_name": a.config_name, "args": vars(a), "n": len(traces),
            "wall_seconds": round(time.time() - t0, 1), "python": platform.python_version()}
    if a.policy == "hf":
        meta["tokens_per_second"] = {r: round(hub.tokens_per_second(r), 2) for r in hub.stats}
        meta["device"] = str(hub.device)
    Path(a.out + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
