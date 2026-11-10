"""Cost of the open-weights guard versus a closed-model API guard. Every assumption is an
explicit argument and is written into the output, because none of the prices are ours.

Measured inputs come from reports/run_meta.json (tokens/second on the GPU we actually used)
and the guard traces (prompt size, decisions). Prices are ASSUMPTIONS you must set to the
current public list prices before quoting the result.

  python -m eval.cost --gpu-hourly 0.60 --api-in-per-mtok 3.0 --api-out-per-mtok 15.0 --requests-per-day 100000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--gpu-name", default=None, help="defaults to the GPU recorded in run_meta.json")
    ap.add_argument("--gpu-hourly", type=float, required=True, help="ASSUMPTION: USD per GPU-hour")
    ap.add_argument("--api-in-per-mtok", type=float, required=True, help="ASSUMPTION: USD per 1M input tokens")
    ap.add_argument("--api-out-per-mtok", type=float, required=True, help="ASSUMPTION: USD per 1M output tokens")
    ap.add_argument("--requests-per-day", type=int, default=100000, help="ASSUMPTION: guard checks per day")
    ap.add_argument("--prompt-tokens", type=int, default=None, help="guard prompt tokens; default: chars/4 from guard data")
    ap.add_argument("--output-tokens", type=int, default=20)
    ap.add_argument("--utilisation", type=float, default=0.5, help="ASSUMPTION: fraction of GPU time doing useful work")
    a = ap.parse_args()
    rep = Path(a.reports)
    meta = json.loads((rep / "run_meta.json").read_text()) if (rep / "run_meta.json").exists() else {}
    tps = (meta.get("tokens_per_second") or {}).get("guard")
    prompt_tokens = a.prompt_tokens
    if prompt_tokens is None and Path("data/sft/guard_val.jsonl").exists():
        rows = [json.loads(l) for l in open("data/sft/guard_val.jsonl")]
        prompt_tokens = int(sum(sum(len(m["content"]) for m in r["messages"][:2]) for r in rows) / len(rows) / 4)
    out = {"assumptions": {"gpu": a.gpu_name or meta.get("gpu"), "gpu_hourly_usd": a.gpu_hourly,
                           "api_usd_per_mtok_in": a.api_in_per_mtok, "api_usd_per_mtok_out": a.api_out_per_mtok,
                           "requests_per_day": a.requests_per_day, "utilisation": a.utilisation,
                           "prompt_tokens_per_check": prompt_tokens, "output_tokens_per_check": a.output_tokens,
                           "prompt_tokens_method": "characters / 4 over guard validation prompts (estimate)"},
           "measured": {"guard_output_tokens_per_second": tps}}
    api_per_check = (prompt_tokens or 0) / 1e6 * a.api_in_per_mtok + a.output_tokens / 1e6 * a.api_out_per_mtok
    out["api_guard"] = {"usd_per_check": api_per_check, "usd_per_day": api_per_check * a.requests_per_day}
    if tps:
        sec_per_check = a.output_tokens / tps          # unbatched, sequential: a conservative upper bound
        gpu_seconds = sec_per_check * a.requests_per_day / a.utilisation
        gpus = max(1, -(-gpu_seconds // 86400))
        out["self_hosted_guard"] = {"seconds_per_check_unbatched": sec_per_check, "gpus_needed": int(gpus),
                                    "usd_per_day": gpus * 24 * a.gpu_hourly,
                                    "usd_per_check": gpus * 24 * a.gpu_hourly / a.requests_per_day}
        out["ratio_api_over_self_hosted"] = out["api_guard"]["usd_per_day"] / out["self_hosted_guard"]["usd_per_day"]
    else:
        out["self_hosted_guard"] = "not computed: no measured guard throughput in run_meta.json yet"
    out["caveats"] = ["Unbatched single-stream throughput; batching or vLLM would lower self-hosted cost.",
                      "Ignores engineering, on-call and idle capacity beyond the utilisation factor.",
                      "A closed API guard may be more or less accurate; this compares cost only."]
    (rep / "cost_estimate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
