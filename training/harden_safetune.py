"""LoRA fine-tuning of Tiny Aya through SafeTune.

Used twice:
  agent hardening   --method safegrad  (SafeTune's SafeGrad: gradient surgery that keeps the
                                        update from conflicting with an alignment set)
                    --method plain     (plain SFT on the same data: the ablation)
  guard classifier  --method plain --data data/sft/guard_train.jsonl

Why SafeGrad: our hardening data teaches a narrow behaviour (ignore instructions in tool
output). Narrow fine-tunes are known to erode a model's general refusal behaviour. SafeGrad
projects the task gradient away from the direction that would damage an alignment set
(BeaverTails refusals, loaded by SafeTune) and adds `rho` times the alignment gradient.

We call SafeTune's `SafeGradHFTrainer` directly rather than `harden.SafeGradTrainer(...).train()`
because the high-level trainer merges the LoRA adapter and saves a full 6.7 GB checkpoint; we
want to publish the adapter. Tokenisation and LoRA wrapping still go through SafeTune.

  python -m training.harden_safetune --method safegrad --out outputs/hardened-safegrad
  python -m training.harden_safetune --method plain    --out outputs/hardened-plain
  python -m training.harden_safetune --method plain --data data/sft/guard_train.jsonl \
        --val data/sft/guard_val.jsonl --out outputs/guard --epochs 2
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

DEFAULT_MODEL = "CohereLabs/tiny-aya-global"


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def strip_padding(ds):
    """SafeTune pads every row to max_len. Keep only real tokens; we pad per batch instead."""
    from datasets import Dataset
    out = []
    for r in ds:
        keep = [i for i, m in enumerate(r["attention_mask"]) if m == 1]
        out.append({k: [r[k][i] for i in keep] for k in ("input_ids", "attention_mask", "labels")})
    return Dataset.from_list(out)


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        import torch
        n = max(len(f["input_ids"]) for f in feats)
        pad = lambda seq, v: list(seq) + [v] * (n - len(seq))
        return {"input_ids": torch.tensor([pad(f["input_ids"], self.pad_id) for f in feats]),
                "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in feats]),
                "labels": torch.tensor([pad(f["labels"], -100) for f in feats])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--method", choices=["safegrad", "plain"], default="safegrad")
    ap.add_argument("--data", default="data/sft/agent_train.jsonl")
    ap.add_argument("--val", default="data/sft/agent_val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=None, help="default: sized by SafeTune, capped at 2048")
    ap.add_argument("--rho", type=float, default=0.1, help="SafeGrad alignment weight (SafeTune docs: start at 0.1)")
    ap.add_argument("--safety-n", type=int, default=256, help="rows of the alignment (safety) set")
    ap.add_argument("--kl-reference", action="store_true",
                    help="load a frozen copy of the base model as SafeGrad's KL reference (about 2x memory)")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--push", default=None, help="HF repo id to push the adapter to")
    a = ap.parse_args()

    import torch
    import transformers
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    import safetune
    from safetune.runner.utils.data_utils import tokenize_dataset
    from safetune.runner.utils.model_utils import lora_wrap

    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kw = {"device_map": "auto"}
    if a.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    elif torch.cuda.is_available():
        kw["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if a.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model = lora_wrap(model, r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05)
    model.print_trainable_parameters()

    def prep(path):
        rows = load_rows(path)[: a.max_rows]
        ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
        return strip_padding(tokenize_dataset(ds, tok, max_len=a.max_len))

    train_ds = prep(a.data)
    val_ds = prep(a.val) if a.val and Path(a.val).exists() else None
    lens = [len(x) for x in train_ds["input_ids"]]
    print(f"train rows={len(train_ds)} tokens: max={max(lens)} mean={sum(lens) / len(lens):.0f}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args_kw = dict(
        output_dir=str(out / "trainer"), num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=1,          # SafeGrad requires 1
        learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * a.epochs * len(train_ds) / a.batch_size)), logging_steps=10,
        eval_strategy="steps" if val_ds is not None else "no", eval_steps=100,
        save_strategy="no", report_to="none", seed=a.seed,
        bf16=use_bf16, fp16=torch.cuda.is_available() and not use_bf16 and not a.load_in_4bit,
    )
    collator = PadCollator(tok.pad_token_id)
    safety_info = None
    if a.method == "safegrad":
        from torch.utils.data import DataLoader
        from safetune.harden import SafeGradConfig, SafeGradHFTrainer
        from safetune.runner import harden
        _, safety_ds = harden.load_harden_data(a.model, n=a.safety_n)   # BeaverTails refusals
        safety_ds = strip_padding(safety_ds)
        safety_loader = DataLoader(safety_ds, batch_size=a.batch_size, shuffle=True, collate_fn=collator)
        reference = None
        if a.kl_reference:
            reference = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
        safety_info = {"source": "safetune.runner.harden.load_harden_data (BeaverTails)", "rows": len(safety_ds),
                       "mode": "kl_to_frozen_reference" if reference is not None else "cross_entropy_proxy"}
        trainer = SafeGradHFTrainer(model=model, args=SafeGradConfig(**args_kw), train_dataset=train_ds,
                                    eval_dataset=val_ds, data_collator=collator, safety_dataset=safety_loader,
                                    reference_model=reference, rho=a.rho)
    else:
        trainer = Trainer(model=model, args=TrainingArguments(**args_kw), train_dataset=train_ds,
                          eval_dataset=val_ds, data_collator=collator)

    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0
    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    log = trainer.state.log_history
    (out / "training_log.json").write_text(json.dumps(log, indent=2))
    run = {
        "method": a.method, "args": vars(a), "train_rows": len(train_ds),
        "val_rows": len(val_ds) if val_ds is not None else 0, "safety_set": safety_info,
        "train_loss": result.training_loss, "wall_seconds": round(wall, 1),
        "final_eval_loss": next((x["eval_loss"] for x in reversed(log) if "eval_loss" in x), None),
        "versions": {"safetune": getattr(safetune, "__version__", "?"), "transformers": transformers.__version__,
                     "torch": torch.__version__, "python": platform.python_version()},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str))
    (adapter_dir / "signers_doubt_run.json").write_text(json.dumps(run, indent=2, default=str))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tr = [(x["step"], x["loss"]) for x in log if "loss" in x]
        ev = [(x["step"], x["eval_loss"]) for x in log if "eval_loss" in x]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(*zip(*tr), label="train loss")
        if ev:
            ax.plot(*zip(*ev), marker="o", label="validation loss")
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title(f"{Path(a.out).name} ({a.method})"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "training_curve.png", dpi=150)
    except Exception as e:                       # plotting must never lose a finished run
        print("plot skipped:", e)
    print(json.dumps({k: run[k] for k in ("method", "train_rows", "train_loss", "final_eval_loss", "wall_seconds")}, indent=2))

    if a.push:
        from safetune import push_to_hub
        push_to_hub(str(adapter_dir), a.push, private=False,
                    commit_message=f"Signer's Doubt {a.method} adapter (SafeTune)")
        print("pushed adapter ->", a.push)


if __name__ == "__main__":
    main()
