"""Keep run outputs on the Hub between Kaggle/Colab sessions.

  python scripts/hub_sync.py push --user Sykeclonex     # runs/, reports/, training logs, adapters
  python scripts/hub_sync.py pull --user Sykeclonex
Work-in-progress files live in the dataset repo <user>/signers-doubt-eval-traces; adapters go to
their own model repos (the same ones the final publish step writes cards for).
"""
import argparse
from pathlib import Path

MARK = Path(".hub_sync_pulled_at")
ADAPTERS = {"outputs/hardened-safegrad": "signers-doubt-tiny-aya-hardened",
            "outputs/hardened-plain": "signers-doubt-tiny-aya-plain-sft",
            "outputs/guard": "signers-doubt-tiny-aya-guard"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["push", "pull"])
    ap.add_argument("--user", required=True)
    a = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    store = f"{a.user}/signers-doubt-eval-traces"
    if a.mode == "push":
        api.create_repo(store, repo_type="dataset", private=False, exist_ok=True)
        # Only files written since the last pull are pushed, so two sessions running different
        # stages cannot overwrite each other's newer results with stale pulled copies.
        since = float(MARK.read_text()) if MARK.exists() else 0.0
        for d in ("runs", "reports"):
            fresh = [str(f.relative_to(d)) for f in Path(d).rglob("*") if f.is_file() and f.stat().st_mtime > since] \
                if Path(d).exists() else []
            if fresh:
                api.upload_folder(folder_path=d, path_in_repo=d, repo_id=store, repo_type="dataset",
                                  commit_message=f"sync {d}", allow_patterns=fresh, ignore_patterns=["*.pt", "*.safetensors", "**/trainer/**", "phase0_*/**", "ARTICLE.md", "abstract.md", ".gitkeep", "phase0_ci/**"])
        for out, name in ADAPTERS.items():
            for f in ("training_log.json", "run.json", "training_curve.png"):
                if (Path(out) / f).exists():
                    api.upload_file(path_or_fileobj=str(Path(out) / f), path_in_repo=f"training/{Path(out).name}/{f}",
                                    repo_id=store, repo_type="dataset")
            if (Path(out) / "adapter/adapter_config.json").exists():
                api.create_repo(f"{a.user}/{name}", repo_type="model", private=False, exist_ok=True)
                api.upload_folder(folder_path=str(Path(out) / "adapter"), repo_id=f"{a.user}/{name}",
                                  repo_type="model", commit_message="adapter")
        print("pushed to", store)
    else:
        try:
            snapshot_download(store, repo_type="dataset", local_dir=".", allow_patterns=["runs/**", "reports/**"])
        except Exception as e:
            print("no stored runs yet:", str(e)[:120])
        MARK.write_text(str(__import__("time").time()))
        for out, name in ADAPTERS.items():
            try:
                snapshot_download(f"{a.user}/{name}", local_dir=str(Path(out) / "adapter"))
                print("pulled adapter", name)
            except Exception:
                pass


if __name__ == "__main__":
    main()
