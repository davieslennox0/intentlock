"""General-capability sanity check through AuditKIT, before and after hardening.

A small MMLU slice (multiple choice) and a small GSM8K slice (arithmetic word problems),
loaded with AuditKIT's built-in scenarios and scored with its `exact_match` metric. The
same seeded sample is used for every model so the before/after comparison is paired, and
`ak.compare` gives the bootstrap significance test.

  python -m eval.capability --out reports/capability --adapter agent=outputs/hardened-safegrad/adapter
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

LETTERS = "ABCD"
MMLU_SUBJECTS = ["high_school_mathematics", "computer_security", "formal_logic", "high_school_macroeconomics"]


def mcq_samples(n_per_subject: int, seed: int):
    import auditkit as ak
    from auditkit.scenarios.mmlu import MMLUScenario
    rng, out = random.Random(seed), []
    for subj in MMLU_SUBJECTS:
        rows = list(MMLUScenario(subject=subj, split="test").samples())
        rng.shuffle(rows)
        for s in rows[:n_per_subject]:
            opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(s.choices))
            out.append(ak.Sample(input=f"{s.input}\n{opts}\n\nAnswer with a single letter (A, B, C or D).",
                                 target=LETTERS[int(s.target)], metadata={"subject": subj}))
    return out


def gsm_samples(n: int, seed: int):
    import auditkit as ak
    from auditkit.scenarios.gsm8k import GSM8KScenario
    rows = list(GSM8KScenario(split="test").samples())
    random.Random(seed).shuffle(rows)
    out = []
    for s in rows[:n]:
        gold = str(s.target).split("####")[-1].strip().replace(",", "")
        out.append(ak.Sample(input=f"{s.input}\n\nSolve it, then give the final number on the last line as: Answer: <number>",
                             target=gold))
    return out


def extract_letter(text: str) -> str:
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else ""


def extract_number(text: str) -> str:
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", text)
    nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
    if not nums:
        return ""
    v = nums[-1].replace(",", "")
    return v[:-2] if v.endswith(".0") else v


def make_model(hub, adapter, extract, max_new_tokens):
    def model(prompts):
        return [extract(hub.generate([{"role": "user", "content": p}], adapter=adapter,
                                     max_new_tokens=max_new_tokens, role="capability")) for p in prompts]
    model.__name__ = f"tiny_aya_{adapter or 'base'}"
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--mmlu-per-subject", type=int, default=50)
    ap.add_argument("--gsm8k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="reports/capability")
    ap.add_argument("--load-in-4bit", action="store_true")
    a = ap.parse_args()
    import auditkit as ak
    from testbed.policies import ModelHub

    hub = ModelHub(a.model, load_in_4bit=a.load_in_4bit)
    names = [None]
    for spec in a.adapter:
        name, path = spec.split("=", 1)
        hub.add_adapter(name, path)
        names.append(name)
    tasks = {"mmlu_subset": (mcq_samples(a.mmlu_per_subject, a.seed), extract_letter, 8),
             "gsm8k_subset": (gsm_samples(a.gsm8k, a.seed), extract_number, 320)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table, runs = {}, {}
    for task, (samples, extract, mnt) in tasks.items():
        for name in names:
            label = name or "base"
            res = ak.evaluate(samples, make_model(hub, name, extract, mnt), ["exact_match"],
                              experiment_name=f"signers-doubt-{task}", tags=[label])
            res.save(str(out / f"{task}__{label}.json"))
            runs[(task, label)] = res
            table.setdefault(task, {})[label] = {"exact_match": res.to_dict()["headline"].get("exact_match"),
                                                 "n": len(samples), "fingerprint": res.to_dict()["fingerprint"]}
        for name in names[1:]:
            try:
                cmp_ = ak.compare(runs[(task, "base")], runs[(task, name)], metric="exact_match")
                table[task][name]["vs_base"] = cmp_.summary() if hasattr(cmp_, "summary") else str(cmp_)
            except Exception as e:
                table[task][name]["vs_base"] = f"compare failed: {e}"
    (out / "capability.json").write_text(json.dumps(table, indent=2, default=str))
    print(json.dumps(table, indent=2, default=str))


if __name__ == "__main__":
    main()
