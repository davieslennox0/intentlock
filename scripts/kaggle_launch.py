"""Launch one pipeline stage as a private Kaggle GPU kernel, under its own run id.

  python scripts/kaggle_launch.py baseline --kaggle-user K --hf-user H
  python scripts/kaggle_launch.py train    --kaggle-user K --hf-user H --from <baseline_run_id>
  python scripts/kaggle_launch.py final    --kaggle-user K --hf-user H --from <baseline_run_id>
  python scripts/kaggle_launch.py status   --kaggle-user K        # project kernels and their state
  python scripts/kaggle_launch.py cancel   --kaggle-user K --slug intentlock-...   # deletes that kernel

Rules this launcher enforces
  * One run, one path. The run id is <UTC time>-<stage>-<commit7>; the kernel is named after it, and
    its outputs go to runs/<run_id>/ and reports/<run_id>/ on the Hub. Nothing writes to a shared path.
  * No launch while another project kernel is queued or running (pushing does NOT cancel a running
    Kaggle kernel). Cancel it first: the only API for that is deleting the kernel.
  * Earlier runs are read-only inputs (runs/inputs/<run_id>/). Traces are only reused when their
    testbed hash matches the code being run; the runner drops the rest and reruns them.
The HF token is read inside Kaggle from the private dataset <kaggle-user>/intentlock-secrets.
"""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

QLORA = "--load-in-4bit --lora-r 16 --lora-alpha 32"
HAND = ("python -m testbed.runner --scenarios data/seed/handwritten.jsonl --out runs/hand/base_seed.jsonl "
        "--config-name base && python -m eval.metrics --traces runs/hand/base_seed.jsonl --out runs/hand/metrics")
SMOKE_TRAIN = [
    f"python -m training.harden_safetune --method safegrad {QLORA} --batch-size 1 --max-len 1024 --longest-first --max-rows 12 --max-steps 12 --safety-n 16 --val '' --out runs/smoke_train/safegrad_1024",
    f"python -m training.harden_safetune --method safegrad {QLORA} --batch-size 1 --max-len 1280 --longest-first --max-rows 12 --max-steps 12 --safety-n 16 --val '' --out runs/smoke_train/safegrad_1280",
    f"python -m training.harden_safetune --method safegrad {QLORA} --batch-size 1 --max-len 1536 --longest-first --max-rows 12 --max-steps 12 --safety-n 16 --val '' --out runs/smoke_train/safegrad_1536",
    f"python -m training.harden_safetune --method plain {QLORA} --batch-size 1 --grad-accum 16 --max-len 1280 --longest-first --max-rows 32 --max-steps 2 --val '' --out runs/smoke_train/plain_1280_accum16",
    "rm -rf runs/smoke_train/*/adapter runs/smoke_train/*/trainer",
]
ALL = "--agent-adapter outputs/hardened-safegrad/adapter --guard-adapter outputs/guard/adapter --plain-adapter outputs/hardened-plain/adapter --no-sampled"

# stage -> lanes. Each lane runs its commands in order on its own GPU. {src} = first --from run id.
STAGES = {
    "baseline": [
        [HAND,
         "python -m eval.run_all --only base base_boundary --splits test --reports reports",
         "python -m eval.run_all --only base_sampled --splits test --reports runs/sampled_reports"],
        ["python -m eval.run_all --only base base_boundary --splits heldout_format --reports reports",
         "python -m eval.run_all --only base --splits train --runs runs/train --reports runs/train_reports",
         "python -m training.build_sft"] + SMOKE_TRAIN,
    ],
    "smoke_train": [["python -m training.build_sft"] + SMOKE_TRAIN],
    "train": [
        ["python -m training.build_sft",
         "python -m training.harden_safetune --method safegrad {train_args} --out outputs/hardened-safegrad || python -m training.harden_safetune --method safegrad {train_args} --amp bf16 --out outputs/hardened-safegrad",
         "python -m training.harden_safetune --method plain {train_args} --out outputs/hardened-plain || python -m training.harden_safetune --method plain {train_args} --amp bf16 --out outputs/hardened-plain"],
        ["python -m training.guard_data --traces runs/inputs/{src}/train/base_train.jsonl",
         "python -m training.harden_safetune --method plain {guard_args} --data data/sft/guard_train.jsonl --val data/sft/guard_val.jsonl --out outputs/guard || python -m training.harden_safetune --method plain {guard_args} --amp bf16 --data data/sft/guard_train.jsonl --val data/sft/guard_val.jsonl --out outputs/guard"],
    ],
    "final": [
        ["mkdir -p runs/final && cp runs/inputs/{src}/final/base*_test.jsonl runs/final/ 2>/dev/null; true",
         f"python -m eval.run_all {ALL} --splits test --reports reports",
         "python -m eval.run_all --agent-adapter outputs/hardened-safegrad/adapter --only base_sampled hardened_sampled --splits test --reports runs/sampled_reports",
         "python -m eval.report --reports reports"],
        ["mkdir -p runs/final && cp runs/inputs/{src}/final/base*_heldout_format.jsonl runs/final/ 2>/dev/null; true",
         f"python -m eval.run_all {ALL} --splits heldout_format --reports reports",
         "python -m eval.capability --out reports/capability --adapter hardened=outputs/hardened-safegrad/adapter --adapter plain_sft=outputs/hardened-plain/adapter"],
    ],
    "interp": [
        ["CUDA_VISIBLE_DEVICES= timeout 5400 python -m interp.circuitkit_discover --compact-system --n 12 --max-tokens 760 --algorithm eap --device cpu --dtype float32 --out reports/interp || CUDA_VISIBLE_DEVICES= timeout 7200 python -m interp.circuitkit_discover --compact-system --n 8 --max-tokens 760 --algorithm eap --device cpu --dtype bfloat16 --out reports/interp || echo 'CircuitKIT discovery failed or timed out; causal_eval falls back to probe-selected layers'",
         "python -m interp.causal_eval --limit 120 --out reports/interp"],
    ],
}

JOB = r"""
import json, os, subprocess, sys, threading, time
STAGE, LANES, HF_USER, COMMIT, RUN_ID, SOURCES = {stage!r}, {lanes!r}, {hf_user!r}, {commit!r}, {run_id!r}, {sources!r}

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
for root, _, files in os.walk("/kaggle/input"):
    if "hf_token.txt" in files:
        token = open(os.path.join(root, "hf_token.txt")).read().strip()
if not token:
    sys.exit("HF token not found: attach the intentlock-secrets dataset")
os.environ.update(HF_TOKEN=token, HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                  PYTHONUNBUFFERED="1", SD_COMMIT=COMMIT, SD_RUN_ID=RUN_ID)

print("RUN", RUN_ID)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; python --version")
os.chdir("/kaggle/working")
sh("rm -rf intentlock && git clone -q https://github.com/davieslennox0/intentlock.git", check=True)
os.chdir("intentlock")
sh(f"git checkout -q {{COMMIT}} && git log --oneline -1", check=True)
os.makedirs("runs/logs", exist_ok=True)
sh("pip uninstall -y -q torchao; pip install -q -r requirements.txt > runs/logs/pip_install.log 2>&1; tail -5 runs/logs/pip_install.log; pip check 2>&1 | head -15; pip freeze > runs/logs/pip_freeze.txt")
src = " ".join(SOURCES)
sh(f"python scripts/hub_sync.py index --user {{HF_USER}} --run-id {{RUN_ID}} --stage {{STAGE}} --status running --from {{src}}")
if SOURCES:
    sh(f"python scripts/hub_sync.py pull --user {{HF_USER}} --from {{src}}")

lock, failed, stop = threading.Lock(), [], threading.Event()
def push():
    with lock:
        subprocess.run(f"python scripts/hub_sync.py push --user {{HF_USER}} --run-id {{RUN_ID}} 2>&1 | tail -1", shell=True)
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
threading.Thread(target=heartbeat, daemon=True).start()
[t.start() for t in threads]; [t.join() for t in threads]
stop.set(); push()
status = "failed" if failed else "complete"
sh(f"python scripts/hub_sync.py index --user {{HF_USER}} --run-id {{RUN_ID}} --stage {{STAGE}} --status {{status}}")
print("\nSTAGE", STAGE, RUN_ID, status, "failed commands:", json.dumps(failed, indent=1))
for i in range(len(LANES)):
    sh(f"tail -40 runs/logs/{{STAGE}}_lane{{i}}.log")
sh("cd /kaggle/working && tar czf intentlock_out.tgz intentlock/reports intentlock/runs 2>/dev/null; rm -rf intentlock")
sys.exit(1 if failed else 0)
"""


def project_kernels(user):
    out = subprocess.run(["kaggle", "kernels", "list", "--mine", "--page-size", "50", "--csv"],
                         capture_output=True, text=True).stdout
    slugs = [l.split(",")[0] for l in out.splitlines()[1:] if "/intentlock-" in l.split(",")[0]]
    res = {}
    for s in slugs:
        st = subprocess.run(["kaggle", "kernels", "status", s], capture_output=True, text=True).stdout
        res[s] = st.strip().split("status")[-1].strip().strip('"').replace("KernelWorkerStatus.", "")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + ["status", "cancel"])
    ap.add_argument("--kaggle-user", required=True)
    ap.add_argument("--hf-user")
    ap.add_argument("--from", dest="sources", nargs="*", default=[], help="run ids whose outputs this stage reads")
    ap.add_argument("--slug", help="cancel: kernel slug to delete")
    ap.add_argument("--train-args", default=f"{QLORA} --batch-size 1 --max-len 1536 --epochs 1 --lr 4e-5 --time-budget-min 170")
    ap.add_argument("--guard-args", default=f"{QLORA} --batch-size 1 --grad-accum 16 --max-len 1024 --epochs 2 --time-budget-min 200")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    kernels = project_kernels(a.kaggle_user)
    if a.stage == "status":
        print(json.dumps(kernels, indent=1))
        return
    if a.stage == "cancel":
        subprocess.run(["kaggle", "kernels", "delete", f"{a.kaggle_user}/{a.slug}", "-y"], check=True)
        return
    live = {k: v for k, v in kernels.items() if v in ("RUNNING", "QUEUED")}
    if live and not a.dry_run:
        raise SystemExit(f"refusing to launch: project kernels still alive: {live}. "
                         f"Cancel with: kaggle_launch.py cancel --kaggle-user {a.kaggle_user} --slug <name>")
    if STAGES[a.stage] and any("{src}" in c for lane in STAGES[a.stage] for c in lane) and not a.sources:
        raise SystemExit(f"stage {a.stage} needs --from <run id of the run it reads from>")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip() and not a.dry_run:
        raise SystemExit("working tree is dirty: commit and push first, so the kernel runs exactly this code")
    if "origin/" not in subprocess.run(["git", "branch", "-r", "--contains", commit], capture_output=True, text=True).stdout and not a.dry_run:
        raise SystemExit("HEAD is not pushed: the kernel clones from GitHub")
    run_id = f"{time.strftime('%Y%m%dT%H%M', time.gmtime())}-{a.stage}-{commit[:7]}"
    slug = f"intentlock-{run_id.lower().replace('_', '-')}"
    fmt = dict(src=a.sources[0] if a.sources else "", train_args=a.train_args, guard_args=a.guard_args)
    lanes = [[c.format(**fmt) if ("{src}" in c or "_args}" in c) else c for c in lane] for lane in STAGES[a.stage]]
    code = JOB.format(stage=a.stage, lanes=lanes, hf_user=a.hf_user, commit=commit, run_id=run_id, sources=a.sources)
    meta = {"id": f"{a.kaggle_user}/{slug}", "title": slug, "code_file": "job.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{a.kaggle_user}/intentlock-secrets"], "competition_sources": [],
            "kernel_sources": [], "model_sources": []}
    d = Path(tempfile.mkdtemp(prefix="kaggle_"))
    (d / "job.py").write_text(code)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    compile(code, "job.py", "exec")
    print(f"run_id={run_id}\nkernel={a.kaggle_user}/{slug}")
    if a.dry_run:
        print(json.dumps(lanes, indent=1))
        return
    subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], check=True)


if __name__ == "__main__":
    main()
