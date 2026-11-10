"""Publish everything to Hugging Face as public, un-gated repos, and build the two collections.

  python scripts/publish_hf.py --user <hf-username> [--what dataset traces models space collections]

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
datasets: [{user}/signers-doubt-scenarios]
---

# {title}

{summary}

LoRA adapter for [`{base}`](https://huggingface.co/{base}), built for **Signer's Doubt** (Lexsi AI
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
    "hardened": ("outputs/hardened-safegrad", "signers-doubt-tiny-aya-hardened", "Signer's Doubt: hardened wallet agent (SafeGrad)",
                 "Tiny Aya fine-tuned to treat tool output as data: it completes the user's task and ignores instructions planted in token metadata, listings, API responses and READMEs.",
                 "SafeTune `SafeGradHFTrainer` (gradient surgery against a BeaverTails alignment set) on safe trajectories built from the `train` split only.", ""),
    "plain": ("outputs/hardened-plain", "signers-doubt-tiny-aya-plain-sft", "Signer's Doubt: plain-SFT ablation",
              "Ablation: the same data and hyperparameters as the hardened adapter, with ordinary SFT instead of SafeGrad.",
              "Plain supervised fine-tuning on safe trajectories built from the `train` split only.", ""),
    "guard": ("outputs/guard", "signers-doubt-tiny-aya-guard", "Signer's Doubt: pre-signing guard",
              "Tiny Aya classifier that reads the user's task, recent tool outputs and a proposed transaction payload, and answers `ALLOW: reason` or `BLOCK: reason`.",
              "Plain SFT on proposed actions from the `train` split: reference safe/compromised trajectories, baseline-agent traces, and up-weighted legitimate large actions.",
              "and `training/guard.py` "),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--what", nargs="+", default=["dataset", "traces", "models", "space", "collections"])
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    u, done = a.user, {"models": [], "datasets": [], "spaces": []}

    if "dataset" in a.what:
        repo = f"{u}/signers-doubt-scenarios"
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for f in ("train.jsonl", "test.jsonl", "heldout_format.jsonl", "curation_report.json",
                      "curatorkit_manifest.json", "curatorkit_lexsi_provenance.json", "curatorkit_checksums.txt"):
                if (ROOT / "data/curated" / f).exists():
                    shutil.copy(ROOT / "data/curated" / f, t / f)
            shutil.copy(ROOT / "data/DATASET_CARD.md", t / "README.md")
            api.upload_folder(folder_path=str(t), repo_id=repo, repo_type="dataset", commit_message="Signer's Doubt scenarios")
        done["datasets"].append(repo)

    if "traces" in a.what:
        repo = f"{u}/signers-doubt-eval-traces"
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        card = ("---\nlicense: cc-by-4.0\ntags: [ai-safety, prompt-injection, agents, eval-logs]\n---\n\n"
                "# Signer's Doubt: evaluation traces and logs\n\nFull agent traces (messages, tool calls, proposed and executed "
                "actions, guard verdicts, labels), AuditKIT reports, training logs and interpretability results for "
                f"[Signer's Doubt](https://github.com/davieslennox0/intentlock). Scenarios: `{u}/signers-doubt-scenarios`.\n\n"
                "`runs/final/<config>_<split>.jsonl` holds one JSON trace per scenario. `reports/` holds the result tables, "
                "plots, AuditKIT outputs, capability benchmark runs and `run_meta.json` (hardware, versions, throughput).\n")
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "README.md").write_text(card)
            for d in ("runs/final", "runs/interp", "reports"):
                if (ROOT / d).exists():
                    shutil.copytree(ROOT / d, t / d)
            for out in ("outputs/hardened-safegrad", "outputs/hardened-plain", "outputs/guard"):
                for f in ("training_log.json", "run.json", "training_curve.png"):
                    if (ROOT / out / f).exists():
                        (t / "training" / Path(out).name).mkdir(parents=True, exist_ok=True)
                        shutil.copy(ROOT / out / f, t / "training" / Path(out).name / f)
            api.upload_folder(folder_path=str(t), repo_id=repo, repo_type="dataset", commit_message="Evaluation traces and logs")
        done["datasets"].append(repo)

    if "models" in a.what:
        for key, (out, name, title, summary, training, extra) in MODELS.items():
            adapter = ROOT / out / "adapter"
            if not adapter.exists():
                print(f"skip {key}: {adapter} not found")
                continue
            repo = f"{u}/{name}"
            run = (ROOT / out / "run.json").read_text() if (ROOT / out / "run.json").exists() else "{}"
            (adapter / "README.md").write_text(MODEL_CARD.format(base=BASE, user=u, title=title, summary=summary,
                                                                 training=training, run=run, repo=repo, extra_use=extra))
            api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
            api.upload_folder(folder_path=str(adapter), repo_id=repo, repo_type="model", commit_message=title)
            done["models"].append(repo)

    if "space" in a.what:
        repo = f"{u}/signers-doubt"
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
                        f = ROOT / f"runs/final/{cfg}_{split}.jsonl"
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
            api.upload_folder(folder_path=str(t), repo_id=repo, repo_type="space", commit_message="Signer's Doubt demo")
        for k, v in (("SD_AGENT_ADAPTER", f"{u}/signers-doubt-tiny-aya-hardened"), ("SD_GUARD_ADAPTER", f"{u}/signers-doubt-tiny-aya-guard")):
            api.add_space_variable(repo, k, v)
        done["spaces"].append(repo)
        print(f"Space created. Add the HF_TOKEN secret at https://huggingface.co/spaces/{repo}/settings")

    if "collections" in a.what:
        for title, kind, items in (
            ("Signer's Doubt: models", "model", [f"{u}/{m[1]}" for m in MODELS.values()]),
            ("Signer's Doubt: datasets", "dataset", [f"{u}/signers-doubt-scenarios", f"{u}/signers-doubt-eval-traces"]),
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
