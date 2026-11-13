"""Publish everything to Hugging Face as public, un-gated repos, and build the two collections.

  python scripts/publish_hf.py --user <hf-username> --final-run <id> --baseline-run <id> --train-run <id> \
      --runs-dir <local copy of runs/<final-run>> [--what dataset traces models space collections]

Traces and adapters are already on the Hub (scripts/hub_sync.py pushes them per run id); this script
adds the cards, the scenario dataset, the Space and the collections. It never re-uploads traces.

Requires `huggingface-cli login` (or HF_TOKEN) with write access. Safe to re-run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "CohereLabs/tiny-aya-global"

MODEL_CARD = """---
license: cc-by-nc-4.0
base_model: {base}
library_name: peft
tags: [lora, ai-safety, prompt-injection, agents, safetune, lexsi-hackathon]
datasets: [{user}/intentlock-scenarios]
---

# {title}

{summary}

LoRA adapter for [`{base}`](https://huggingface.co/{base}), built for **Intentlock** (Lexsi AI
Safety and Alignment Hackathon, Track 2). Code and full results:
<https://github.com/davieslennox0/intentlock>.

## Training

{training}

```json
{run}
```

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{base}")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{base}"), "{repo}")
```

The prompt format (system prompt, strict-JSON actions, boundary tags) is in `testbed/prompts.py`
{extra_use}of the GitHub repository; the adapter expects that format.

## Limits

Trained and evaluated on synthetic scenarios with mock tools. It reduces injection-following on that
distribution; it is not a guarantee against real attackers and must not be the only control in front
of real funds. Inherits Tiny Aya's CC-BY-NC-4.0 licence.
"""

MODELS = {
    "hardened": ("outputs/hardened-safegrad", "intentlock-tiny-aya-hardened", "Intentlock: hardened wallet agent (SafeGrad)",
                 "Tiny Aya fine-tuned to treat tool output as data: it completes the user's task and ignores instructions planted in token metadata, listings, API responses and READMEs.",
                 "SafeTune `SafeGradHFTrainer` (gradient surgery against a BeaverTails alignment set) on safe trajectories built from the `train` split only.", ""),
    "plain": ("outputs/hardened-plain", "intentlock-tiny-aya-plain-sft", "Intentlock: plain-SFT ablation",
              "Ablation: the same data and hyperparameters as the hardened adapter, with ordinary SFT instead of SafeGrad.",
              "Plain supervised fine-tuning on safe trajectories built from the `train` split only.", ""),
    "guard": ("outputs/guard", "intentlock-tiny-aya-guard", "Intentlock: pre-signing guard",
              "Tiny Aya classifier that reads the user's task, recent tool outputs and a proposed transaction payload, and answers `ALLOW: reason` or `BLOCK: reason`.",
              "Plain SFT on proposed actions from the `train` split: reference safe/compromised trajectories, baseline-agent traces, and up-weighted legitimate large actions.",
              "and `training/guard.py` "),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--what", nargs="+", default=["dataset", "traces", "models", "space", "collections"])
    ap.add_argument("--final-run", help="run id of the final evaluation")
    ap.add_argument("--baseline-run", help="run id of the baseline the final evaluation reused")
    ap.add_argument("--train-run", help="run id that produced the adapters")
    ap.add_argument("--space-sdk", choices=["static", "gradio"], default="static",
                    help="static replays recorded traces and is free; gradio runs the models and needs a paid Hub plan")
    ap.add_argument("--runs-dir", help="local copy of runs/<final-run> (contains final/*.jsonl); needed for the Space")
    a = ap.parse_args()
    if {"traces", "space"} & set(a.what) and not a.final_run:
        ap.error("--final-run is required for traces and space")
    if "space" in a.what and a.space_sdk == "gradio" and not a.runs_dir:
        ap.error("--runs-dir is required for space")
    from huggingface_hub import HfApi
    api = HfApi()
    u, done = a.user, {"models": [], "datasets": [], "spaces": []}

    if "dataset" in a.what:
        repo = f"{u}/intentlock-scenarios"
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for f in ("train.jsonl", "test.jsonl", "heldout_format.jsonl", "curation_report.json",
                      "curatorkit_manifest.json", "curatorkit_lexsi_provenance.json", "curatorkit_checksums.txt"):
                if (ROOT / "data/curated" / f).exists():
                    shutil.copy(ROOT / "data/curated" / f, t / f)
            shutil.copy(ROOT / "data/DATASET_CARD.md", t / "README.md")
            api.upload_folder(folder_path=str(t), repo_id=repo, repo_type="dataset", commit_message="Intentlock scenarios")
        done["datasets"].append(repo)

    if "traces" in a.what:
        repo = f"{u}/intentlock-eval-traces"
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        card = ("---\nlicense: cc-by-4.0\ntags: [ai-safety, prompt-injection, agents, eval-logs]\n---\n\n"
                "# Intentlock: evaluation traces and logs\n\nFull agent traces (messages, tool calls, proposed and executed "
                "actions, guard verdicts, labels), AuditKIT reports, training logs and interpretability results for "
                f"[Intentlock](https://github.com/davieslennox0/intentlock). Scenarios: `{u}/intentlock-scenarios`.\n\n"
                "Every run writes under its own id: `runs/<run_id>/` holds traces and logs, `reports/<run_id>/` holds result "
                "tables, plots, AuditKIT outputs, capability runs and `run_meta*.json` (hardware, versions, throughput). "
                "`RUNS.json` indexes all runs. Each trace records the code commit and testbed hash that produced it.\n\n"
                "## Runs behind the published numbers\n\n| Role | Run id |\n|---|---|\n"
                f"| Final evaluation | `{a.final_run}` |\n| Baseline (base and prompt-rule traces reused by the final run) | `{a.baseline_run}` |\n"
                f"| Training (adapters, guard data) | `{a.train_run}` |\n\n"
                f"`runs/{a.final_run}/final/<config>_<split>.jsonl` holds one JSON trace per scenario. Other runs listed in "
                "`RUNS.json` are earlier or superseded and are kept for the record.\n")
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                        commit_message="Dataset card: layout and canonical runs")
        done["datasets"].append(repo)

    if "models" in a.what:
        for key, (out, name, title, summary, training, extra) in MODELS.items():
            repo = f"{u}/{name}"
            if not api.repo_exists(repo):
                print(f"skip {key}: {repo} not on the Hub")
                continue
            try:  # hub_sync stores the training record next to the adapter
                from huggingface_hub import hf_hub_download
                run = Path(hf_hub_download(repo, "signers_doubt_run.json")).read_text()
            except Exception:
                run = "{}"
            card = MODEL_CARD.format(base=BASE, user=u, title=title, summary=summary, training=training, run=run,
                                     repo=repo, extra_use=extra)
            api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="model",
                            commit_message="Model card")
            done["models"].append(repo)

    if "space" in a.what and a.space_sdk == "static":
        # Free tier: a static Space that replays recorded traces (the same page as the project site).
        repo = f"{u}/intentlock"
        api.create_repo(repo, repo_type="space", space_sdk="static", private=False, exist_ok=True)
        readme = ("---\ntitle: Intentlock\nemoji: 🖋️\ncolorFrom: indigo\ncolorTo: green\nsdk: static\n"
                  "app_file: index.html\npinned: false\nlicense: mit\n"
                  "short_description: Base vs hardened wallet agent on poisoned tool output\n---\n\n"
                  "Replay demo for [Intentlock](https://github.com/davieslennox0/intentlock): the same poisoned tool "
                  "output given to base Tiny Aya and to the hardened agent with a pre-signing guard. Episodes are "
                  f"replayed from recorded traces of run `{a.final_run}`; no model runs in this Space. All tools are "
                  "mocks; nothing is signed, sent or executed. A live Gradio version is in `space/` in the repository.\n")
        api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo, repo_type="space",
                        commit_message="Intentlock replay demo")
        for f in ("index.html", "data.json"):
            api.upload_file(path_or_fileobj=str(ROOT / "site" / f), path_in_repo=f, repo_id=repo, repo_type="space",
                            commit_message="Intentlock replay demo")
        done["spaces"].append(repo)

    if "space" in a.what and a.space_sdk == "gradio":
        repo = f"{u}/intentlock"
        api.create_repo(repo, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for f in ("app.py", "requirements.txt", "README.md", "examples.jsonl"):
                shutil.copy(ROOT / "space" / f, t / f)
            # Recorded traces for the examples, so the free CPU Space can replay them.
            ids = {json.loads(l)["id"] for l in open(ROOT / "space/examples.jsonl")}
            rec = {}
            for side, cfgs in (("base", ["base"]), ("hardened", ["hardened_guard", "hardened"])):
                for cfg in cfgs:
                    for split in ("test", "heldout_format"):
                        f = Path(a.runs_dir) / f"final/{cfg}_{split}.jsonl"
                        for l in (open(f) if f.exists() else []):
                            tr = json.loads(l)
                            if tr["scenario_id"] in ids:
                                rec.setdefault(tr["scenario_id"], {}).setdefault(side, tr)
            rec = {k: v for k, v in rec.items() if len(v) == 2}
            (t / "recorded.json").write_text(json.dumps(rec))
            print(f"space: recorded traces for {len(rec)}/{len(ids)} examples")
            shutil.copytree(ROOT / "testbed", t / "testbed", ignore=shutil.ignore_patterns("__pycache__"))
            (t / "training").mkdir()
            (t / "training/__init__.py").write_text("")
            shutil.copy(ROOT / "training/guard.py", t / "training/guard.py")
            api.upload_folder(folder_path=str(t), repo_id=repo, repo_type="space", commit_message="Intentlock demo")
        for k, v in (("SD_AGENT_ADAPTER", f"{u}/intentlock-tiny-aya-hardened"), ("SD_GUARD_ADAPTER", f"{u}/intentlock-tiny-aya-guard")):
            api.add_space_variable(repo, k, v)
        done["spaces"].append(repo)
        print(f"Space created. Add the HF_TOKEN secret at https://huggingface.co/spaces/{repo}/settings")

    if "collections" in a.what:
        for title, kind, items in (
            ("Intentlock: models", "model", [f"{u}/{m[1]}" for m in MODELS.values()]),
            ("Intentlock: datasets", "dataset", [f"{u}/intentlock-scenarios", f"{u}/intentlock-eval-traces"]),
        ):
            col = api.create_collection(title=title, namespace=u, private=False, exists_ok=True,
                                        description="Lexsi AI Safety and Alignment Hackathon, Track 2: wallet-agent prompt injection.")
            for item in items:
                try:
                    api.add_collection_item(col.slug, item_id=item, item_type=kind, exists_ok=True)
                except Exception as e:
                    print(f"  could not add {item}: {str(e)[:120]}")
            print("collection:", f"https://huggingface.co/collections/{col.slug}")
    print(json.dumps(done, indent=2))


if __name__ == "__main__":
    main()
