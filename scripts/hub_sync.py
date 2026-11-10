"""Keep run outputs on the Hub. Every run writes under its own run id and nothing is shared.

  Hub layout (dataset repo <user>/signers-doubt-eval-traces)
    runs/<run_id>/...      traces and logs of that run        (local: runs/)
    reports/<run_id>/...   tables, plots, AuditKIT output      (local: reports/)
    RUNS.json              index: run id -> stage, commit, testbed hash, status, inputs

  push --run-id R            upload local runs/ and reports/ to runs/R/ and reports/R/
  pull --from R1 [R2 ...]    download earlier runs into local runs/ and reports/ (read as inputs)
  index --run-id R --stage S --status ...   record or update the run in RUNS.json

Run ids look like 20261110T1305-baseline-4f5111e. Adapters go to their own model repos.
"""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

ADAPTERS = {"outputs/hardened-safegrad": "signers-doubt-tiny-aya-hardened",
            "outputs/hardened-plain": "signers-doubt-tiny-aya-plain-sft",
            "outputs/guard": "signers-doubt-tiny-aya-guard"}
IGNORE = ["*.pt", "*.safetensors", "**/trainer/**", "phase0_*/**", "ARTICLE.md", "abstract.md", ".gitkeep",
          "phase0_ci/**", "inputs/**"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["push", "pull", "index"])
    ap.add_argument("--user", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--from", dest="sources", nargs="*", default=[])
    ap.add_argument("--stage")
    ap.add_argument("--status", default="running")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    store = f"{a.user}/signers-doubt-eval-traces"

    if a.mode == "push":
        assert a.run_id, "push needs --run-id"
        api.create_repo(store, repo_type="dataset", private=False, exist_ok=True)
        for d in ("runs", "reports"):
            if Path(d).exists() and any(Path(d).rglob("*")):
                api.upload_folder(folder_path=d, path_in_repo=f"{d}/{a.run_id}", repo_id=store, repo_type="dataset",
                                  commit_message=f"{a.run_id}: sync {d}", ignore_patterns=IGNORE)
        for out, name in ADAPTERS.items():
            for f in ("training_log.json", "run.json", "training_curve.png"):
                if (Path(out) / f).exists():
                    api.upload_file(path_or_fileobj=str(Path(out) / f), repo_id=store, repo_type="dataset",
                                    path_in_repo=f"runs/{a.run_id}/training/{Path(out).name}/{f}")
            if (Path(out) / "adapter/adapter_config.json").exists() and (Path(out) / "run.json").exists():
                api.create_repo(f"{a.user}/{name}", repo_type="model", private=False, exist_ok=True)
                api.upload_folder(folder_path=str(Path(out) / "adapter"), repo_id=f"{a.user}/{name}",
                                  repo_type="model", commit_message=f"adapter from run {a.run_id}")
        print("pushed", a.run_id)

    elif a.mode == "pull":
        # Earlier runs are inputs: they land in runs/inputs/<run_id>/ and reports/inputs/<run_id>/,
        # which are never pushed back and never written to by this run.
        for rid in a.sources:
            tmp = Path(tempfile.mkdtemp())
            try:
                snapshot_download(store, repo_type="dataset", local_dir=str(tmp),
                                  allow_patterns=[f"runs/{rid}/**", f"reports/{rid}/**"])
            except Exception as e:
                print(f"could not pull {rid}: {str(e)[:120]}")
                continue
            for d in ("runs", "reports"):
                src = tmp / d / rid
                if src.exists():
                    dst = Path(d) / "inputs" / rid
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(src, dst, dirs_exist_ok=True)
            print("pulled", rid)
        for out, name in ADAPTERS.items():
            try:
                snapshot_download(f"{a.user}/{name}", local_dir=str(Path(out) / "adapter"))
                print("pulled adapter", name)
            except Exception:
                pass

    else:
        assert a.run_id and a.stage
        api.create_repo(store, repo_type="dataset", private=False, exist_ok=True)
        try:
            idx = json.load(open(hf_hub_download(store, "RUNS.json", repo_type="dataset", force_download=True)))
        except Exception:
            idx = {}
        import os
        rec = idx.get(a.run_id, {})
        rec.update(stage=a.stage, status=a.status, updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   code_commit=os.environ.get("SD_COMMIT", rec.get("code_commit")), inputs=a.sources or rec.get("inputs", []))
        if a.note:
            rec["note"] = a.note
        try:
            from testbed.agent import TESTBED_HASH
            rec["testbed_hash"] = TESTBED_HASH
        except Exception:
            pass
        idx[a.run_id] = rec
        api.upload_file(path_or_fileobj=json.dumps(idx, indent=2).encode(), path_in_repo="RUNS.json",
                        repo_id=store, repo_type="dataset", commit_message=f"{a.run_id}: {a.status}")
        print("indexed", a.run_id, a.status)


if __name__ == "__main__":
    main()
