"""Rename the Hugging Face repos from signers-doubt-* to intentlock-*. Run once, with a write token:

  HF_TOKEN=... python scripts/hf_rename.py --user <hf-username> [--dry-run]

Refuses while RUNS.json lists a run as running, because a live run that pushes to the old name
could recreate it and split the data. The Hub keeps redirects from the old names.
"""
import argparse
import json

PAIRS = [("dataset", "signers-doubt-eval-traces", "intentlock-eval-traces"),
         ("dataset", "signers-doubt-scenarios", "intentlock-scenarios"),
         ("model", "signers-doubt-tiny-aya-hardened", "intentlock-tiny-aya-hardened"),
         ("model", "signers-doubt-tiny-aya-plain-sft", "intentlock-tiny-aya-plain-sft"),
         ("model", "signers-doubt-tiny-aya-guard", "intentlock-tiny-aya-guard"),
         ("space", "signers-doubt", "intentlock")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="rename even if RUNS.json lists a running run")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    try:
        runs = json.load(open(hf_hub_download(f"{a.user}/signers-doubt-eval-traces", "RUNS.json",
                                              repo_type="dataset", force_download=True)))
        live = [k for k, r in runs.items() if isinstance(r, dict) and r.get("status") == "running"]
    except Exception as e:
        print("could not read RUNS.json:", str(e)[:100]); live = []
    if live and not a.force:
        raise SystemExit(f"runs still marked running: {live}; wait, or pass --force")
    for kind, old, new in PAIRS:
        src, dst = f"{a.user}/{old}", f"{a.user}/{new}"
        if api.repo_exists(dst, repo_type=kind):
            print(f"ok      {kind:8} {dst} already exists"); continue
        if not api.repo_exists(src, repo_type=kind):
            print(f"absent  {kind:8} {src} (nothing to rename)"); continue
        print(f"{'would move' if a.dry_run else 'move'}    {kind:8} {src} -> {dst}")
        if not a.dry_run:
            api.move_repo(from_id=src, to_id=dst, repo_type=kind)


if __name__ == "__main__":
    main()
