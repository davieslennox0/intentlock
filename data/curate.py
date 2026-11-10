"""Curate raw scenarios with CuratorKIT, then apply the consistency gate.

  python -m data.curate --raw data/raw/scenarios.jsonl --out data/curated

Stage 1  CuratorKIT: schema gate, exact + MinHash near-duplicate removal, text cleaning.
         Each scenario is flattened to (instruction=user task, input=tool fixtures,
         output=safe final answer) so the dedup sees what the agent sees.
Stage 2  data.validate: malicious_action_spec / expected_actions consistency and a
         replay of the safe and compromised reference trajectories.
Stage 3  Leakage check between train and test/heldout, balanced down-sampling to the
         target sizes, and per-split JSONL export.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .validate import validate

ID_TAG = re.compile(r"\[scenario (sd-\d+)\]")
TARGET = {"attack": 800, "clean": 300}
FRACS = {"train": 0.60, "test": 0.25, "heldout_format": 0.15}


def flatten(sc: dict) -> dict:
    fixtures = "\n".join(f"[{k}] {v}" for k, v in sc["tool_fixtures"].items())
    return {"instruction": sc["user_task"], "input": f"[scenario {sc['id']}]\n{fixtures}",
            "output": sc.get("safe_final", "")}


def run_curatorkit(rows: list[dict], workdir: Path, threshold: float) -> tuple[set[str], dict]:
    from curatorkit import Curator, CuratorConfig
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / "flattened.jsonl"
    with open(src, "w") as f:
        for sc in rows:
            f.write(json.dumps(flatten(sc), ensure_ascii=False) + "\n")
    out = workdir / "curatorkit_out"
    if out.exists():
        shutil.rmtree(out)
    Curator(CuratorConfig(
        dataset=str(src), dedup="minhash", minhash_threshold=threshold, clean=True,
        dedup_ignore_patterns=[r"\[scenario sd-\d+\]"],   # the id tag must not defeat dedup
        min_tokens=5, max_tokens=2048, export_formats=["alpaca"], output_dir=str(out),
        name="signers_doubt_scenarios",
    )).run()
    kept = set()
    with open(out / "sft_alpaca.jsonl") as f:
        for line in f:
            m = ID_TAG.search(json.loads(line).get("input", ""))
            if m:
                kept.add(m.group(1))
    return kept, json.loads((out / "manifest.json").read_text())


def injection_key(sc: dict) -> str:
    """Injection text with addresses/numbers/tokens masked: the phrasing template."""
    t = sc.get("injected_text", "")
    t = re.sub(r"0x[0-9a-fA-F]{40}", "<ADDR>", t)
    t = re.sub(r"[A-Za-z0-9+/=]{24,}", "<BLOB>", t)
    t = re.sub(r"\b\d[\d.,]*\b", "<N>", t)
    t = re.sub(r"\b[A-Z]{3,5}\b", "<SYM>", t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/scenarios.jsonl")
    ap.add_argument("--out", default="data/curated")
    ap.add_argument("--minhash-threshold", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    rows = [json.loads(l) for l in open(a.raw) if l.strip()]
    report = {"raw": len(rows)}

    kept_ids, manifest = run_curatorkit(rows, out, a.minhash_threshold)
    report["curatorkit_stage_counts"] = manifest.get("stage_counts")
    report["curatorkit_version"] = manifest.get("tool_versions", {}).get("curatorkit")
    after_ck = [sc for sc in rows if sc["id"] in kept_ids]
    report["after_curatorkit"] = len(after_ck)

    rejected, valid = [], []
    for sc in after_ck:
        errs = validate(sc)
        (rejected if errs else valid).append((sc, errs))
    report["consistency_rejected"] = len(rejected)
    report["consistency_reasons"] = dict(Counter(e for _, errs in rejected for e in errs))
    valid = [sc for sc, _ in valid]

    # Leakage: an exact fixture set must not appear in both train and an eval split.
    train_fx = {json.dumps(sc["tool_fixtures"], sort_keys=True) for sc in valid if sc["split"] == "train"}
    leak = [sc["id"] for sc in valid if sc["split"] != "train"
            and json.dumps(sc["tool_fixtures"], sort_keys=True) in train_fx]
    valid = [sc for sc in valid if sc["id"] not in set(leak)]
    report["leak_removed"] = len(leak)
    # Heldout phrasing templates must be unseen in train.
    train_keys = {injection_key(sc) for sc in valid if sc["split"] == "train" and not sc["is_clean"]}
    shared = [sc["id"] for sc in valid if sc["split"] == "heldout_format" and not sc["is_clean"]
              and sc["injection_technique"] not in ("encoded_base64", "encoded_hex")
              and injection_key(sc) in train_keys]
    report["heldout_templates_seen_in_train"] = len(shared)

    # Down-sample to target sizes, keeping the family mix.
    by = defaultdict(list)
    for sc in valid:
        by[(sc["family"], sc["split"])].append(sc)
    fam_total = Counter(sc["family"] for sc in valid if not sc["is_clean"])
    n_attack = sum(fam_total.values())
    final = []
    for (family, split), group in sorted(by.items()):
        rng.shuffle(group)
        if family == "clean":
            want = round(TARGET["clean"] * FRACS[split])
        else:
            want = round(TARGET["attack"] * fam_total[family] / n_attack * FRACS[split])
        final.extend(group[:want])
    final.sort(key=lambda s: s["id"])

    for split in FRACS:
        with open(out / f"{split}.jsonl", "w") as f:
            for sc in final:
                if sc["split"] == split:
                    f.write(json.dumps(sc, ensure_ascii=False) + "\n")
    with open(out / "rejected.jsonl", "w") as f:
        for sc, errs in rejected:
            f.write(json.dumps({"id": sc["id"], "reasons": errs}) + "\n")
    report["final"] = len(final)
    report["final_attack"] = sum(not s["is_clean"] for s in final)
    report["final_clean"] = sum(s["is_clean"] for s in final)
    report["by_family_split"] = {f"{k[0]}/{k[1]}": v for k, v in
                                 sorted(Counter((s["family"], s["split"]) for s in final).items())}
    report["by_technique"] = dict(Counter(s["injection_technique"] for s in final))
    report["by_location"] = dict(Counter(s["injection_location"] for s in final))
    report["by_wrapper"] = dict(Counter(s["wrapper"] for s in final))
    (out / "curation_report.json").write_text(json.dumps(report, indent=2))
    for f in ("manifest.json", "lexsi_provenance.json", "checksums.txt"):
        src = out / "curatorkit_out" / f
        if src.exists():
            shutil.copy(src, out / f"curatorkit_{f}")
    (out / "flattened.jsonl").unlink(missing_ok=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
