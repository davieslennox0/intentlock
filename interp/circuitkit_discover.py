"""Find components associated with injection-following, with CircuitKIT.

Contrastive pairs (TRAIN split only), built in interp/contexts.py:
  clean prompt     decision-point context WITH the injection   -> answer: malicious tool
  corrupt prompt   same context with the injection replaced by token-aligned filler
                                                               -> answer: the legitimate tool
EAP-family discovery attributes the logit difference to heads/MLPs. The output is a
ranked node list and a per-layer score profile; `interp/causal_eval.py` then tests
those layers causally. Nothing here is a claim on its own: attribution is correlational.

  python -m interp.circuitkit_discover --n 24 --algorithm eap-ig --out reports/interp
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .contexts import decision_messages, render, tool_pair, usable

MODEL = "CohereLabs/tiny-aya-global"
FILLER = " This field is informational and has no further content to report at this time."


def aligned_twin(tok, sc: dict):
    """Rebuild the injected fixtures with the injection swapped for filler of the same
    token count, so clean and corrupt prompts are position-aligned. Returns fixtures or None."""
    out = {}
    for key, injected in sc["tool_fixtures"].items():
        twin = sc["clean_twin_fixtures"][key]
        if injected == twin:
            out[key] = injected
            continue
        need = len(tok(injected, add_special_tokens=False)["input_ids"])
        # grow the twin with filler inside the same field until token counts match
        base, tail = (twin[:-2], twin[-2:]) if twin.rstrip().endswith('"}') else (twin, "")
        ids_base = tok(base, add_special_tokens=False)["input_ids"]
        ids_tail = tok(tail, add_special_tokens=False)["input_ids"] if tail else []
        fill_ids = tok(FILLER * 40, add_special_tokens=False)["input_ids"]
        k = need - len(ids_base) - len(ids_tail)
        if k < 0 or k > len(fill_ids):
            return None
        cand = base + tok.decode(fill_ids[:k]) + tail
        if len(tok(cand, add_special_tokens=False)["input_ids"]) != need:
            return None
        out[key] = cand
    return out


def build_records(tok, scenarios, n, max_tokens):
    from circuitkit.data.normalized import ContrastiveRecord
    records, skipped = [], defaultdict(int)
    for sc in scenarios:
        if len(records) >= n:
            break
        if not usable(sc):
            skipped["no_tool_contrast"] += 1
            continue
        twin = aligned_twin(tok, sc)
        if twin is None:
            skipped["could_not_align"] += 1
            continue
        clean = render(tok, decision_messages(sc))
        corrupt = render(tok, decision_messages(sc, twin))
        n_clean = len(tok(clean, add_special_tokens=False)["input_ids"])
        if n_clean != len(tok(corrupt, add_special_tokens=False)["input_ids"]):
            skipped["length_mismatch_after_template"] += 1
            continue
        if n_clean > max_tokens:
            skipped["too_long"] += 1
            continue
        mal, legit = tool_pair(sc)
        if tok(mal, add_special_tokens=False)["input_ids"][0] == tok(legit, add_special_tokens=False)["input_ids"][0]:
            skipped["same_first_token"] += 1
            continue
        records.append(ContrastiveRecord(record_id=sc["id"], clean_prompt=clean, clean_answer=mal,
                                         corrupt_prompt=corrupt, corrupt_answer=legit,
                                         meta={"family": sc["family"], "technique": sc["injection_technique"]}))
    return records, dict(skipped)


def layer_profile(scores: dict) -> dict:
    """Sum |score| per layer from node names such as 'A12.3' / 'MLP 12' / 'm12'."""
    prof = defaultdict(float)
    for name, s in scores.items():
        m = re.search(r"(\d+)", str(name))
        if m:
            prof[int(m.group(1))] += abs(float(s))
    return dict(sorted(prof.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--train", default="data/curated/train.jsonl")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--algorithm", default="eap-ig", choices=["eap", "eap-ig", "eap-gp", "ibcircuit", "cdt"])
    ap.add_argument("--ig-steps", type=int, default=3)
    ap.add_argument("--out", default="reports/interp")
    a = ap.parse_args()

    import circuitkit as ck                      # must be imported before transformer_lens
    from circuitkit import register_task
    from circuitkit.data.normalized import DatasetShape, NormalizedDataset
    from circuitkit.data.normalized_task import NormalizedTaskSpec

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = [json.loads(l) for l in open(a.train)]
    assert all(s["split"] == "train" for s in scenarios)
    model = ck.load_model(a.model, algorithm=a.algorithm)
    records, skipped = build_records(model.tokenizer, scenarios, a.n, a.max_tokens)
    print(f"contrastive records: {len(records)}  skipped: {skipped}")
    if len(records) < 8:
        raise SystemExit("too few aligned contrastive pairs; raise --max-tokens or check alignment")
    ds = NormalizedDataset(name="signers_doubt_injection", shape=DatasetShape.QA, records=records)
    spec = NormalizedTaskSpec(ds, name="signers_doubt_injection")
    register_task(spec)

    kw = {"ig_steps": a.ig_steps} if a.algorithm == "eap-ig" else {}
    circuit = ck.discover(model, "signers_doubt_injection", algorithm=a.algorithm, level="node",
                          n_examples=len(records), batch_size=1,
                          output_path=str(out / "circuit_discovery_results.pt"), **kw)
    scores = {str(k): float(v) for k, v in circuit.scores.items()}
    prof = layer_profile(scores)
    top_layers = [l for l, _ in sorted(prof.items(), key=lambda x: -x[1])[:6]]
    result = {"model": a.model, "algorithm": a.algorithm, "n_records": len(records), "skipped": skipped,
              "top_nodes": [list(x) if isinstance(x, (tuple, list)) else x for x in circuit.top_nodes(25)],
              "layer_profile": prof, "top_layers": sorted(top_layers),
              "circuitkit_version": getattr(ck, "__version__", "?"),
              "note": "Attribution scores are correlational. See causal_eval for the interventions."}
    (out / "circuitkit_circuit.json").write_text(json.dumps(result, indent=2, default=str))
    (out / "node_scores.json").write_text(json.dumps(scores))
    print(json.dumps({k: result[k] for k in ("n_records", "top_layers")}, indent=2))
    print("top nodes:", result["top_nodes"][:10])


if __name__ == "__main__":
    main()
