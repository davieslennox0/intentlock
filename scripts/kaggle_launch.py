"""Launch one pipeline stage as a private Kaggle GPU kernel.

  python scripts/kaggle_launch.py baseline --kaggle-user <kaggle-user> --hf-user Sykeclonex

The kernel clones this repo at the given commit, pulls earlier outputs from the Hub, runs the
stage, and pushes its outputs back to the Hub, so stages chain across sessions. The HF token is
read inside Kaggle from a private dataset `<kaggle-user>/intentlock-secrets` (file hf_token.txt)
or a Kaggle secret named HF_TOKEN; it is never written into the kernel source.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

STAGES = {
    "smoke": ["for c in transformers auditkit curatorkit safetune circuitkit; do python scripts/phase0_smoke.py $c | tail -45; done; true"],
    "baseline": [
        "python -m eval.run_all --only base base_boundary",
        "python -m eval.run_all --only base_sampled --splits test",
        "python -m eval.run_all --only base --splits train --runs runs/train --reports runs/train_reports",
        "python -m eval.capability --out reports/capability_base",
    ],
    "train": [
        "python -m training.build_sft",
        "python -m training.guard_data --traces runs/train/base_train.jsonl",
        "python -m training.harden_safetune --method safegrad --out outputs/hardened-safegrad --epochs 2",
        "python scripts/hub_sync.py push --user {hf_user}",
        "python -m training.harden_safetune --method plain --data data/sft/guard_train.jsonl --val data/sft/guard_val.jsonl --out outputs/guard --epochs 2",
        "python scripts/hub_sync.py push --user {hf_user}",
        "python -m training.harden_safetune --method plain --out outputs/hardened-plain --epochs 2",
    ],
    "final": [
        "python -m eval.run_all --agent-adapter outputs/hardened-safegrad/adapter --guard-adapter outputs/guard/adapter --plain-adapter outputs/hardened-plain/adapter --no-sampled",
        "python scripts/hub_sync.py push --user {hf_user}",
        "python -m eval.run_all --agent-adapter outputs/hardened-safegrad/adapter --only base_sampled hardened_sampled --splits test",
        "python -m eval.capability --out reports/capability --adapter hardened=outputs/hardened-safegrad/adapter --adapter plain_sft=outputs/hardened-plain/adapter",
        "python -m eval.report --reports reports",
    ],
    "interp": [
        "python -m interp.circuitkit_discover --n 24 --algorithm eap-ig --out reports/interp || echo 'CircuitKIT discovery failed; causal_eval falls back to probe-selected layers'",
        "python -m interp.causal_eval --limit 120 --out reports/interp",
    ],
}

JOB = r'''
import json, os, subprocess, sys, time
STAGE, COMMANDS, HF_USER, COMMIT, KAGGLE_USER = {stage!r}, {commands!r}, {hf_user!r}, {commit!r}, {kaggle_user!r}

def sh(cmd, check=False):
    print("\n$", cmd, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit {{r.returncode}} in {{time.time() - t0:.0f}}s]", flush=True)
    if check and r.returncode:
        sys.exit(r.returncode)
    return r.returncode

token = None
p = f"/kaggle/input/intentlock-secrets/hf_token.txt"
if os.path.exists(p):
    token = open(p).read().strip()
else:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("no Kaggle secret:", e)
if not token:
    sys.exit("HF token not found: attach the intentlock-secrets dataset or add a Kaggle secret HF_TOKEN")
os.environ.update(HF_TOKEN=token, HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")

sh("nvidia-smi --query-gpu=name,memory.total --format=csv; python --version")
os.chdir("/kaggle/working")
sh("rm -rf intentlock && git clone -q https://github.com/davieslennox0/intentlock.git", check=True)
os.chdir("intentlock")
sh(f"git checkout -q {{COMMIT}} && git log --oneline -1", check=True)
sh("pip uninstall -y -q torchao; pip install -q -r requirements.txt 2>&1 | tail -4; pip check 2>&1 | head -15")
sh(f"python scripts/hub_sync.py pull --user {{HF_USER}}")
failed = []
for cmd in COMMANDS:
    if sh(cmd):
        failed.append(cmd)
    sh(f"python scripts/hub_sync.py push --user {{HF_USER}} | tail -1")
print("\nSTAGE", STAGE, "finished. failed commands:", json.dumps(failed, indent=1))
sh("rm -rf /kaggle/working/intentlock/outputs/*/trainer; cd /kaggle/working && tar czf intentlock_reports.tgz intentlock/reports intentlock/runs 2>/dev/null; rm -rf intentlock")
sys.exit(1 if failed else 0)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--kaggle-user", required=True)
    ap.add_argument("--hf-user", required=True)
    ap.add_argument("--commit", default=None, help="default: current HEAD (must be pushed)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    commit = a.commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    cmds = [c.format(hf_user=a.hf_user) for c in STAGES[a.stage]]
    code = JOB.format(stage=a.stage, commands=cmds, hf_user=a.hf_user, commit=commit, kaggle_user=a.kaggle_user)
    meta = {"id": f"{a.kaggle_user}/intentlock-{a.stage}", "title": f"intentlock-{a.stage}", "code_file": "job.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [f"{a.kaggle_user}/intentlock-secrets"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    d = Path(tempfile.mkdtemp(prefix="kaggle_"))
    (d / "job.py").write_text(code)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    compile(code, "job.py", "exec")
    print(f"stage={a.stage} commit={commit[:8]} dir={d}")
    if a.dry_run:
        print(code[:1500])
        return
    subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], check=True)


if __name__ == "__main__":
    main()
