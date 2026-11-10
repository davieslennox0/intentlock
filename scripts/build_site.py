"""Build site/data.json for the static demo from real run outputs. No model numbers are invented:
anything not yet produced by a run is left null and the page says "pending".

  python scripts/build_site.py --baseline <dir with final/*.jsonl and results_*.json> [--final <dir>]
A "dir" is a local copy of runs/<run_id>/ plus reports/<run_id>/ merged, or the repo root after a run.
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP_STEP = ("tool", "args", "output", "final", "parse_error", "payload", "guard", "thought")
CONFIGS = ["base", "base_boundary", "plain_sft", "hardened", "base_guard", "hardened_guard"]
METRICS = ["asr", "exposure_rate", "asr_given_exposed", "task_completion_under_attack", "task_completion_clean",
           "false_block_rate", "asr_tag_spoof", "parse_error_rate"]


def slim(trace):
    steps = []
    for s in trace["steps"]:
        d = {k: s[k] for k in KEEP_STEP if s.get(k) not in (None, "")}
        if "output" in d:
            d["output"] = d["output"][:1200]
        if "payload" in d:
            d["payload"] = {k: v for k, v in d["payload"].items() if v not in ("", None) and k not in ("data", "step")}
        if "guard" in d:
            d["guard"] = {k: d["guard"].get(k) for k in ("verdict", "reason")}
        steps.append(d)
    return {"steps": steps, "labels": trace["labels"], "stop_reason": trace["stop_reason"],
            "exposed": trace.get("exposed_to_injection"), "code_commit": trace.get("code_commit", "")[:7]}


def find(dirs, pattern):
    for d in dirs:
        hits = sorted(Path(d).rglob(pattern)) if d and Path(d).exists() else []
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="*", default=[], help="directories holding the baseline run's files")
    ap.add_argument("--final", nargs="*", default=[], help="directories holding the final-eval run's files")
    ap.add_argument("--baseline-run", default=None)
    ap.add_argument("--final-run", default=None)
    ap.add_argument("--out", default=str(ROOT / "site/data.json"))
    a = ap.parse_args()
    examples = [json.loads(l) for l in open(ROOT / "space/examples.jsonl")]
    ids = {e["id"] for e in examples}
    traces = {i: {} for i in ids}
    for side, dirs, cfgs in (("base", a.final + a.baseline, ["base"]), ("hardened", a.final, ["hardened_guard", "hardened"])):
        for cfg in cfgs:
            for split in ("test", "heldout_format"):
                f = find(dirs, f"{cfg}_{split}.jsonl")
                for line in (open(f) if f else []):
                    t = json.loads(line)
                    if t["scenario_id"] in ids and side not in traces[t["scenario_id"]]:
                        traces[t["scenario_id"]][side] = dict(slim(t), config=cfg)
    results = {}
    for split in ("test", "heldout_format"):
        merged = {}
        for d in (a.baseline, a.final):                     # final overrides baseline for shared configs
            f = find(d, f"results_{split}.json")
            if f:
                merged.update(json.loads(f.read_text()))
        if merged:
            results[split] = {c: {m: merged[c].get(m) for m in METRICS} | {
                "asr_by_family": merged[c].get("asr_by_family"), "n_attack": merged[c].get("n_attack"),
                "n_clean": merged[c].get("n_clean"), "guard": merged[c].get("guard")} for c in CONFIGS if c in merged}
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "runs": {"baseline": a.baseline_run, "final": a.final_run},
        "examples": [{k: e.get(k) for k in ("id", "family", "split", "user_task", "tool_fixtures", "injection_location",
                                            "injection_technique", "attack_goal", "malicious_action_spec",
                                            "expected_actions", "is_clean", "wrapper", "injected_text")}
                     | {"traces": traces[e["id"]]} for e in examples],
        "results": results,
    }
    Path(a.out).write_text(json.dumps(data, ensure_ascii=False))
    n_b = sum("base" in t for t in traces.values()); n_h = sum("hardened" in t for t in traces.values())
    print(f"wrote {a.out}: {len(examples)} examples, base traces {n_b}, hardened traces {n_h}, results for {list(results)}")


if __name__ == "__main__":
    main()
