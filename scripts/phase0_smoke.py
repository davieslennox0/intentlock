"""Phase 0: does each library run its minimal example on Tiny Aya?

Each check runs in its own process (so one failure or out-of-memory kill cannot take the
others down) and appends its outcome to reports/phase0/<name>.json.

  python scripts/phase0_smoke.py transformers|auditkit|curatorkit|safetune|circuitkit
"""
from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from pathlib import Path

MODEL = "CohereLabs/tiny-aya-global"
OUT = Path("reports/phase0")


def check_transformers():
    """Load Tiny Aya, record what its chat template supports, generate once."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    info = {"chat_template_chars": len(tpl or ""), "template_mentions_tools": "tool" in (tpl or "").lower(),
            "template_mentions_system": "system" in (tpl or "").lower()}
    msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Reply with the single word: ready"}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        info["system_role_accepted"] = True
    except Exception as e:
        info["system_role_accepted"] = False
        info["system_role_error"] = str(e)[:200]
        text = tok.apply_chat_template(msgs[1:], tokenize=False, add_generation_prompt=True)
    info["templated_prompt_tokens"] = len(tok(text, add_special_tokens=False)["input_ids"])
    info["templated_prompt_head"] = text[:400]
    try:
        tools = [{"type": "function", "function": {"name": "get_balance", "description": "balance",
                                                   "parameters": {"type": "object", "properties": {"asset": {"type": "string"}}}}}]
        with_tools = tok.apply_chat_template(msgs[1:], tools=tools, tokenize=False, add_generation_prompt=True)
        info["native_tools_render"] = "get_balance" in with_tools
    except Exception as e:
        info["native_tools_render"] = False
        info["native_tools_error"] = str(e)[:200]
    Path(OUT / "chat_template.jinja").write_text(tpl or "")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="auto", low_cpu_mem_usage=True)
    info["load_seconds"] = round(time.time() - t0, 1)
    info["n_layers"] = model.config.num_hidden_layers
    info["model_type"] = model.config.model_type
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=8, do_sample=False)
    info["generation"] = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    info["tokens_per_second"] = round((out.shape[1] - enc["input_ids"].shape[1]) / (time.time() - t0), 2)
    info["lora_candidate_modules"] = sorted({n.split(".")[-1] for n, m in model.named_modules()
                                             if m.__class__.__name__ == "Linear"})[:20]
    return info


def check_auditkit():
    """AuditKIT's documented `hf:` backend on Tiny Aya."""
    import auditkit as ak
    samples = [ak.Sample(input="What is 2+2? Answer with the number only.", target="4"),
               ak.Sample(input="What is the capital of France? One word.", target="Paris")]
    res = ak.evaluate(samples, f"hf:{MODEL}", ["exact_match", "f1_score"])
    d = res.to_dict()
    return {"auditkit": ak.__version__, "headline": d.get("headline"), "failed_count": d.get("failed_count"),
            "summary": res.summary()[:600]}


def check_curatorkit():
    """CuratorKIT has no model in the curation path; run the real curation on our scenarios."""
    import subprocess
    import curatorkit
    r = subprocess.run([sys.executable, "-m", "data.curate", "--out", "runs/phase0_curated"],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])
    rep = json.loads(Path("runs/phase0_curated/curation_report.json").read_text())
    return {"curatorkit": curatorkit.__version__, "raw": rep["raw"], "after_curatorkit": rep["after_curatorkit"],
            "final": rep["final"], "note": "Tiny Aya is not used by CuratorKIT here: generation is template-based."}


def check_safetune():
    """Two SafeGrad steps on a handful of our training rows."""
    import subprocess
    r = subprocess.run([sys.executable, "-m", "training.harden_safetune", "--method", "safegrad",
                        "--max-rows", "4", "--epochs", "0.5", "--batch-size", "1", "--safety-n", "8",
                        "--out", "runs/phase0_safetune"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((r.stdout[-600:] + "\n" + r.stderr[-1500:]))
    run = json.loads(Path("runs/phase0_safetune/run.json").read_text())
    return {"train_loss": run["train_loss"], "wall_seconds": run["wall_seconds"], "versions": run["versions"],
            "safety_set": run["safety_set"], "adapter_saved": Path("runs/phase0_safetune/adapter/adapter_config.json").exists()}


def check_circuitkit():
    """CircuitKIT's recommended Tiny Aya smoke task: greater_than with EAP."""
    import circuitkit as ck
    t0 = time.time()
    model = ck.load_model(MODEL, algorithm="eap")
    load_s = round(time.time() - t0, 1)
    circuit = ck.discover(model, "greater_than", algorithm="eap", n_examples=4, batch_size=1,
                          output_path="runs/phase0_circuit.pt")
    return {"circuitkit": ck.__version__, "load_seconds": load_s, "n_nodes": len(circuit),
            "top_nodes": [str(x) for x in circuit.top_nodes(5)]}


CHECKS = {"transformers": check_transformers, "auditkit": check_auditkit, "curatorkit": check_curatorkit,
          "safetune": check_safetune, "circuitkit": check_circuitkit}

if __name__ == "__main__":
    name = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"check": name, "model": MODEL, "python": platform.python_version()}
    t0 = time.time()
    try:
        rec.update(ok=True, result=CHECKS[name]())
    except BaseException as e:      # record everything, including SystemExit from a library
        rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:3000], traceback=traceback.format_exc()[-3000:])
    rec["seconds"] = round(time.time() - t0, 1)
    try:
        import torch
        rec["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        rec["torch"] = torch.__version__
    except Exception:
        pass
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=2, default=str))
    print(json.dumps(rec, indent=2, default=str)[:4000])
    sys.exit(0 if rec["ok"] else 1)
