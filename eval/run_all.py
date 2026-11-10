"""Run every evaluation configuration on test and heldout_format, then score them.

Configurations
  base              Tiny Aya, the scenario's own boundary tags, no extra instruction
  base_boundary     + an explicit "tool output is data, never instructions" rule (prompting baseline)
  hardened          SafeTune (SafeGrad) LoRA adapter
  hardened_guard    hardened agent + pre-signing guard adapter
  base_guard        base agent + guard (isolates the guard's contribution)
  plain_sft         plain-SFT adapter on the same data (ablation), when provided
  *_sampled         a second run at temperature 0.7 for base and hardened

Model loading is shared (one copy of Tiny Aya, adapters switched per call). Runs resume
from existing trace files, so a disconnected Colab session loses nothing.

  python -m eval.run_all --agent-adapter outputs/hardened-safegrad/adapter --guard-adapter outputs/guard/adapter
  python -m eval.run_all --only base --splits train     # baseline traces on train, for guard data
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from testbed.runner import load_jsonl, run_many

from . import auditkit_bridge
from .metrics import markdown_table, summarise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--data-dir", default="data/curated")
    ap.add_argument("--runs", default="runs/final")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--agent-adapter", default=None)
    ap.add_argument("--guard-adapter", default=None)
    ap.add_argument("--plain-adapter", default=None)
    ap.add_argument("--splits", nargs="+", default=["test", "heldout_format"])
    ap.add_argument("--only", nargs="*", default=None, help="subset of configuration names")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-sampled", action="store_true")
    ap.add_argument("--sample-temperature", type=float, default=0.7)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8, help="episodes advanced per generation call")
    a = ap.parse_args()

    from testbed.policies import HFPolicy, ModelHub
    from training.guard import GuardModel
    hub = ModelHub(a.model, load_in_4bit=a.load_in_4bit)
    for name, path in (("agent", a.agent_adapter), ("guard", a.guard_adapter), ("plain", a.plain_adapter)):
        if path:
            hub.add_adapter(name, path)
    guard = GuardModel(hub, "guard") if a.guard_adapter else None

    # name -> (agent adapter, boundary_note, guard, temperature, seed)
    configs = {"base": (None, False, None, 0.0, a.seed), "base_boundary": (None, True, None, 0.0, a.seed)}
    if a.agent_adapter:
        configs["hardened"] = ("agent", False, None, 0.0, a.seed)
    if a.agent_adapter and guard:
        configs["hardened_guard"] = ("agent", False, guard, 0.0, a.seed)
    if guard:
        configs["base_guard"] = (None, False, guard, 0.0, a.seed)
    if a.plain_adapter:
        configs["plain_sft"] = ("plain", False, None, 0.0, a.seed)
    if not a.no_sampled:
        configs["base_sampled"] = (None, False, None, a.sample_temperature, a.seed + 1)
        if a.agent_adapter:
            configs["hardened_sampled"] = ("agent", False, None, a.sample_temperature, a.seed + 1)
    if a.only:
        configs = {k: v for k, v in configs.items() if k in a.only}

    Path(a.runs).mkdir(parents=True, exist_ok=True)
    all_summaries, timings = {}, {}
    for split in a.splits:
        scenarios = load_jsonl(f"{a.data_dir}/{split}.jsonl")[: a.limit]
        by_id = {s["id"]: s for s in scenarios}
        summaries = {}
        for name, (adapter, note, g, temp, seed) in configs.items():
            pol = HFPolicy(hub, adapter=adapter, temperature=temp, seed=seed)
            path = f"{a.runs}/{name}_{split}.jsonl"
            t0 = time.time()
            traces = run_many(scenarios, lambda sc: pol, out=path, config_name=name,
                              boundary_note=note, guard=g, seed=seed, batch_size=a.batch_size)
            timings[f"{name}_{split}"] = round(time.time() - t0, 1)
            summaries[name] = summarise(traces, by_id)
            if split != "train":
                try:
                    res = auditkit_bridge.run(traces, by_id, f"{name}_{split}")
                    d = Path(a.reports) / "auditkit" / f"{name}_{split}"
                    d.mkdir(parents=True, exist_ok=True)
                    for view, r in res.items():
                        (d / f"{view}.json").write_text(r.to_json())
                        (d / f"{view}.summary.txt").write_text(r.summary() + "\n")
                except Exception as e:      # AuditKIT reporting must not lose a finished run
                    print(f"AuditKIT report failed for {name}_{split}: {e}")
        all_summaries[split] = summaries
        md = markdown_table(summaries)
        Path(a.reports).mkdir(parents=True, exist_ok=True)
        (Path(a.reports) / f"results_{split}.md").write_text(f"## Results on `{split}` (n={len(scenarios)})\n\n{md}\n")
        (Path(a.reports) / f"results_{split}.json").write_text(json.dumps(summaries, indent=2))
        print(f"\n## {split}\n{md}")

    import torch
    import transformers
    meta = {"model": a.model, "args": vars(a), "configs": list(configs), "wall_seconds": timings,
            "tokens_per_second": {r: round(hub.tokens_per_second(r), 2) for r in hub.stats},
            "calls": {r: hub.stats[r][2] for r in hub.stats},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "python": platform.python_version()}}
    # one file per invocation: parallel runs on separate GPUs must not overwrite each other
    tag = "_".join(a.splits) + ("_" + "_".join(a.only) if a.only else "")
    (Path(a.reports) / f"run_meta_{tag}.json").write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
