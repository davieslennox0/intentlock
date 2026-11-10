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

SMOKE = "for c in transformers auditkit curatorkit safetune circuitkit; do python scripts/phase0_smoke.py $c | tail -60; done; true"
HAND = "python -m testbed.runner --scenarios data/seed/handwritten.jsonl --out runs/hand/base_seed.jsonl --config-name base && python -m eval.metrics --traces runs/hand/base_seed.jsonl --out runs/hand/metrics"

# stage -> lanes. Each lane is a list of commands run in order on its own GPU (lane index = GPU index).
STAGES = {
    "smoke": [[HAND], [SMOKE]],
    "baseline": [
        [HAND,
         "python -m eval.run_all --only base base_boundary --reports reports",
         "python -m eval.run_all --only base_sampled --splits test --reports runs/sampled_reports"],
        [SMOKE,
         "python -m eval.run_all --only base --splits train --runs runs/train --reports runs/train_reports",
         "python -m eval.capability --out reports/capability_base"],
    ],
    "train": [
        ["python -m training.build_sft",
         "python -m training.harden_safetune --method safegrad --out outputs/hardened-safegrad --load-in-4bit --epochs 2 --batch-size 2 || python -m training.harden_safetune --method safegrad --out outputs/hardened-safegrad --load-in-4bit --epochs 2 --batch-size 1",
         "python -m training.harden_safetune --method plain --out outputs/hardened-plain --load-in-4bit --epochs 2 --batch-size 2 || python -m training.harden_safetune --method plain --out outputs/hardened-plain --load-in-4bit --epochs 2 --batch-size 1"],
        [HAND,
         "python -m eval.run_all --only base --splits train --runs runs/train --reports runs/train_reports",
         "python -m training.guard_data --traces runs/train/base_train.jsonl",
         "python -m training.harden_safetune --method plain --data data/sft/guard_train.jsonl --val data/sft/guard_val.jsonl --out outputs/guard --load-in-4bit --epochs 2 --batch-size 4 || python -m training.harden_safetune --method plain --data data/sft/guard_train.jsonl --val data/sft/guard_val.jsonl --out outputs/guard --load-in-4bit --epochs 2 --batch-size 2",
         "python -m eval.run_all --only base base_boundary --reports reports"],
    ],
    "final": [
        ["python -m eval.run_all --agent-adapter outputs/hardened-safegrad/adapter --guard-adapter outputs/guard/adapter --plain-adapter outputs/hardened-plain/adapter --no-sampled --splits test --reports reports",
         "python -m eval.run_all --agent-adapter outputs/hardened-safegrad/adapter --only hardened_sampled --splits test --reports runs/sampled_reports"],
        ["python -m eval.run_all --agent-adapter outputs/hardened-safegrad/adapter --guard-adapter outputs/guard/adapter --plain-adapter outputs/hardened-plain/adapter --no-sampled --splits heldout_format --reports reports",
         "python -m eval.capability --out reports/capability --adapter hardened=outputs/hardened-safegrad/adapter --adapter plain_sft=outputs/hardened-plain/adapter"],
    ],
    "interp": [
        ["CUDA_VISIBLE_DEVICES= timeout 10800 python -m interp.circuitkit_discover --n 16 --max-tokens 800 --algorithm eap --device cpu --dtype float32 --out reports/interp || echo 'CircuitKIT discovery failed or timed out; causal_eval falls back to probe-selected layers'",
         "python -m interp.causal_eval --limit 120 --out reports/interp"],
    ],
}

JOB = r"""
import json, os, subprocess, sys, threading, time
STAGE, LANES, HF_USER, COMMIT = {stage!r}, {lanes!r}, {hf_user!r}, {commit!r}

def sh(cmd, check=False, env=None, log=None):
    print("\n$", cmd, flush=True)
    t0 = time.time()
    if log:
        with open(log, "a") as f:
            f.write(f"\n$ {{cmd}}\n"); f.flush()
            r = subprocess.run(cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)
            f.write(f"[exit {{r.returncode}} in {{time.time() - t0:.0f}}s]\n")
    else:
        r = subprocess.run(cmd, shell=True, env=env)
    print(f"[exit {{r.returncode}} in {{time.time() - t0:.0f}}s] {{cmd[:80]}}", flush=True)
    if check and r.returncode:
        sys.exit(r.returncode)
    return r.returncode

token = None
if os.path.exists("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        if "hf_token.txt" in files:
            token = open(os.path.join(root, "hf_token.txt")).read().strip()
if not token:
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
os.makedirs("runs/logs", exist_ok=True)
sh("pip uninstall -y -q torchao; pip install -q -r requirements.txt > runs/logs/pip_install.log 2>&1; tail -5 runs/logs/pip_install.log; pip check 2>&1 | head -15; pip freeze > runs/logs/pip_freeze.txt")
sh(f"python scripts/hub_sync.py pull --user {{HF_USER}}")

lock, failed, stop = threading.Lock(), [], threading.Event()
def push():
    with lock:
        subprocess.run(f"python scripts/hub_sync.py push --user {{HF_USER}} 2>&1 | tail -1", shell=True)
def heartbeat():
    while not stop.wait(420):
        push()
def lane(i, cmds):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
    for cmd in cmds:
        if sh(cmd, env=env, log=f"runs/logs/{{STAGE}}_lane{{i}}.log"):
            failed.append(cmd)
        push()
threads = [threading.Thread(target=lane, args=(i, c)) for i, c in enumerate(LANES)]
hb = threading.Thread(target=heartbeat, daemon=True); hb.start()
[t.start() for t in threads]; [t.join() for t in threads]
stop.set(); push()
print("\nSTAGE", STAGE, "finished. failed commands:", json.dumps(failed, indent=1))
for i in range(len(LANES)):
    sh(f"tail -40 runs/logs/{{STAGE}}_lane{{i}}.log")
sh("cd /kaggle/working && tar czf intentlock_out.tgz intentlock/reports intentlock/runs 2>/dev/null; rm -rf intentlock")
sys.exit(1 if failed else 0)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--kaggle-user", required=True)
    ap.add_argument("--hf-user", required=True)
    ap.add_argument("--commit", default=None, help="default: current HEAD (must be pushed)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    commit = a.commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    code = JOB.format(stage=a.stage, lanes=STAGES[a.stage], hf_user=a.hf_user, commit=commit)
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
        print(code[-1700:])
        return
    st = subprocess.run(["kaggle", "kernels", "status", meta["id"]], capture_output=True, text=True).stdout
    if ("RUNNING" in st or "QUEUED" in st) and not a.force:
        raise SystemExit(f"{meta['id']} is still running. Pushing does NOT cancel it; both would run and "
                         "overwrite each other on the Hub. Stop it in the Kaggle UI first, or pass --force.")
    subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], check=True)


if __name__ == "__main__":
    main()
