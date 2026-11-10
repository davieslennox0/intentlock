"""Generate the Colab notebooks from one source of truth. Run: python scripts/make_notebooks.py"""
import json
from pathlib import Path

REPO = "https://github.com/davieslennox0/intentlock.git"
SETUP = [
    ("md", "### Setup\nRuntime → Change runtime type → **T4 GPU** (or better). Accept the Tiny Aya licence at "
           "https://huggingface.co/CohereLabs/tiny-aya-global first: the weights are gated."),
    ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv || echo 'NO GPU: switch the runtime to a GPU'"),
    ("code", f"import os\nif not os.path.exists('intentlock'):\n    !git clone -q {REPO}\n%cd intentlock\n"
             "!pip uninstall -y -q torchao            # SafeTune's Colab note\n"
             "!pip install -q -r requirements.txt 2>&1 | tail -3\n!pip check | head -20"),
    ("code", "from huggingface_hub import login, whoami\nlogin()        # token with read access to Tiny Aya and write access to your namespace\n"
             "HF_USER = whoami()['name']; print(HF_USER)"),
]

NOTEBOOKS = {
    "00_phase0_smoke.ipynb": ("Phase 0: do all four Lexsi libraries run on Tiny Aya?", [
        ("md", "Each check runs in its own process and writes `reports/phase0/<name>.json`."),
        ("code", "for name in ['transformers', 'auditkit', 'curatorkit', 'safetune', 'circuitkit']:\n"
                 "    print('=' * 20, name)\n    !python scripts/phase0_smoke.py {name} | tail -40"),
        ("code", "import json, glob\nfor f in sorted(glob.glob('reports/phase0/*.json')):\n    r = json.load(open(f))\n"
                 "    print(f\"{r['check']:13s} {'OK ' if r['ok'] else 'FAIL'} {r['seconds']:>7}s  {r.get('error', '')[:150]}\")"),
        ("code", "print(open('reports/phase0/chat_template.jinja').read()[:3000])"),
    ]),
    "01_dataset_and_testbed.ipynb": ("Phases 1-2: testbed and dataset (no GPU needed)", [
        ("code", "!python -m pytest -q tests"),
        ("md", "Regenerate and curate. The committed `data/curated/` files should be reproduced exactly."),
        ("code", "!python -m data.generate --seed 7\n!python -m data.curate | tail -45"),
        ("code", "import json, collections\nrows = [json.loads(l) for s in ('train', 'test', 'heldout_format') for l in open(f'data/curated/{s}.jsonl')]\n"
                 "for k, v in sorted(collections.Counter((r['family'], r['split']) for r in rows).items()): print(k, v)"),
        ("md", "Five hand-written scenarios through the base agent, with full traces."),
        ("code", "!python -m testbed.runner --scenarios data/seed/handwritten.jsonl --out runs/hand_base.jsonl --config-name base"),
        ("code", "import json\nfor l in open('runs/hand_base.jsonl'):\n    t = json.loads(l)\n    print('=' * 80); print(t['scenario_id'], t['family'], '| labels:', t['labels'])\n"
                 "    for s in t['steps']: print('  ', s.get('tool') or 'FINAL', json.dumps(s.get('args') or s.get('final') or s.get('raw'))[:200])"),
        ("code", "!python scripts/publish_hf.py --user {HF_USER} --what dataset"),
    ]),
    "02_baseline_hardening_guard.ipynb": ("Phases 3-4: baseline audit, SafeTune hardening, guard", [
        ("md", "### Baseline\n`base` and `base_boundary` on `test` and `heldout_format`, plus a sampled run. Traces resume if the session drops."),
        ("code", "!python -m eval.run_all --only base base_boundary base_sampled"),
        ("md", "**Checkpoint:** if baseline ASR is under 15%, stop here: the attacks are too weak or the agent too passive, and training would show nothing."),
        ("code", "print(open('reports/results_test.md').read()); print(open('reports/results_heldout_format.md').read())"),
        ("md", "### Capability before hardening (AuditKIT: MMLU and GSM8K slices)"),
        ("code", "!python -m eval.capability --out reports/capability_base"),
        ("md", "### Guard data needs the base agent's real mistakes on the *train* split"),
        ("code", "!python -m eval.run_all --only base --splits train --runs runs/train --reports runs/train_reports\n"
                 "!python -m training.build_sft\n!python -m training.guard_data --traces runs/train/base_train.jsonl"),
        ("md", "### 4a. Agent hardening with SafeTune SafeGrad, and the plain-SFT ablation"),
        ("code", "!python -m training.harden_safetune --method safegrad --out outputs/hardened-safegrad --epochs 2"),
        ("code", "!python -m training.harden_safetune --method plain --out outputs/hardened-plain --epochs 2"),
        ("md", "### 4b. Guard"),
        ("code", "!python -m training.harden_safetune --method plain --data data/sft/guard_train.jsonl --val data/sft/guard_val.jsonl --out outputs/guard --epochs 2"),
        ("code", "from IPython.display import Image, display\nfor d in ('hardened-safegrad', 'hardened-plain', 'guard'): display(Image(f'outputs/{d}/training_curve.png'))"),
        ("md", "### Quick test-set numbers before the full evaluation"),
        ("code", "!python -m eval.run_all --only base hardened hardened_guard --splits test --limit 60 --runs runs/quick --reports runs/quick_reports --no-sampled "
                 "--agent-adapter outputs/hardened-safegrad/adapter --guard-adapter outputs/guard/adapter"),
        ("code", "!python scripts/publish_hf.py --user {HF_USER} --what models"),
    ]),
    "03_interp_final_eval_publish.ipynb": ("Phases 5-7: CircuitKIT, final evaluation, publishing", [
        ("md", "If this is a fresh session, pull the adapters back from the Hub."),
        ("code", "from huggingface_hub import snapshot_download\nimport os\n"
                 "for name, out in (('hardened', 'hardened-safegrad'), ('plain-sft', 'hardened-plain'), ('guard', 'guard')):\n"
                 "    if not os.path.exists(f'outputs/{out}/adapter'):\n"
                 "        snapshot_download(f'{HF_USER}/signers-doubt-tiny-aya-{name}', local_dir=f'outputs/{out}/adapter')"),
        ("md", "### Phase 5 (stretch): CircuitKIT discovery, then causal tests\nDiscovery is attribution only. The claim rests on `causal_eval`, which ablates the direction, "
               "compares against a random-direction control, and checks the held-out split. A negative verdict is a result and gets reported."),
        ("code", "!python -m interp.circuitkit_discover --n 24 --algorithm eap-ig --out reports/interp"),
        ("code", "!python -m interp.causal_eval --limit 120 --out reports/interp\nprint(open('reports/interp/causal_results.json').read()[:3000])"),
        ("md", "### Phase 6: every configuration on both evaluation splits"),
        ("code", "!python -m eval.run_all --agent-adapter outputs/hardened-safegrad/adapter --guard-adapter outputs/guard/adapter --plain-adapter outputs/hardened-plain/adapter"),
        ("code", "!python -m eval.capability --out reports/capability --adapter hardened=outputs/hardened-safegrad/adapter --adapter plain_sft=outputs/hardened-plain/adapter"),
        ("code", "!python -m eval.report --reports reports\nprint(open('reports/results_test.md').read()); print(open('reports/results_heldout_format.md').read())"),
        ("md", "Cost comparison. **Set the three prices to current public list prices**; the script records them as assumptions."),
        ("code", "# !python -m eval.cost --gpu-hourly <USD> --api-in-per-mtok <USD> --api-out-per-mtok <USD> --requests-per-day 100000"),
        ("md", "### Phase 7: publish traces, Space and collections"),
        ("code", "!python scripts/publish_hf.py --user {HF_USER} --what traces space collections"),
        ("code", "!cd reports && zip -qr ../reports.zip . && echo 'download reports.zip and commit reports/ to GitHub'"),
    ]),
}


def cell(kind, src):
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


out = Path(__file__).resolve().parents[1] / "notebooks"
for name, (title, cells) in NOTEBOOKS.items():
    nb = {"cells": [cell("md", f"# Intentlock: {title}")] + [cell(k, s) for k, s in SETUP + cells],
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / name).write_text(json.dumps(nb, indent=1))
    print("wrote", name, len(nb["cells"]), "cells")
