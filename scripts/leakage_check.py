"""Train / evaluation leakage check for the Intentlock scenario splits.

Compares `train` against `test` and `heldout_format` at three levels:

  exact      identical user task, identical tool-fixture set, identical single fixture text,
             identical injection string
  template   the same after masking addresses, numbers, long blobs and token symbols
             (i.e. the same phrasing template with different values filled in)
  near-dup   Jaccard similarity of word 3-shingles over the text the agent sees
             (user task + all fixture text), raw and masked; exact Jaccard, not a MinHash estimate

With --traces it also splits attack success by whether a scenario's masked injection string
occurs in train.

  python scripts/leakage_check.py --traces <dir with final/*.jsonl> --out reports/leakage_check.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENCODED = {"encoded_base64", "encoded_hex"}


def mask(t: str) -> str:
    t = re.sub(r"0x[0-9a-fA-F]{6,}", "<ADDR>", t)
    t = re.sub(r"[A-Za-z0-9+/=]{24,}", "<BLOB>", t)
    t = re.sub(r"\b\d[\d.,]*\b", "<N>", t)
    t = re.sub(r"\b[A-Z]{3,5}\b", "<SYM>", t)
    return t


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def visible(sc) -> str:
    return sc["user_task"] + "\n" + "\n".join(v for _, v in sorted(sc["tool_fixtures"].items()))


def shingles(t: str, k=3):
    w = re.findall(r"[a-z0-9<>_]+", t.lower())
    return frozenset(" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1)))


def jac(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/curated"))
    ap.add_argument("--traces", help="directory holding final/<config>_<split>.jsonl")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", default=str(ROOT / "reports/leakage_check.json"))
    a = ap.parse_args()
    D = {s: [json.loads(l) for l in open(Path(a.data) / f"{s}.jsonl")] for s in ("train", "test", "heldout_format")}
    tr = D["train"]
    ids = [set(x["id"] for x in D[s]) for s in D]
    out = {"n": {s: len(D[s]) for s in D}, "threshold": a.threshold,
           "shared_ids": len(ids[0] & ids[1]) + len(ids[0] & ids[2]) + len(ids[1] & ids[2])}

    T = {
        "task": {norm(x["user_task"]) for x in tr},
        "task_masked": {mask(x["user_task"]) for x in tr},
        "fixture_set": {json.dumps(x["tool_fixtures"], sort_keys=True) for x in tr},
        "fixture": {norm(v) for x in tr for v in x["tool_fixtures"].values()},
        "fixture_masked": {mask(v) for x in tr for v in x["tool_fixtures"].values()},
        "inj": {norm(x["injected_text"]) for x in tr if not x["is_clean"]},
        "inj_masked": {mask(x["injected_text"]) for x in tr if not x["is_clean"]},
    }
    tr_raw = [shingles(visible(x)) for x in tr]
    tr_msk = [shingles(mask(visible(x))) for x in tr]
    flags = {}
    for split in ("test", "heldout_format"):
        rows = D[split]
        atk = [x for x in rows if not x["is_clean"]]
        head = [x for x in atk if x["injection_technique"] not in ENCODED]
        r = {"n": len(rows), "n_attack": len(atk), "n_headline": len(head)}
        r["exact"] = {
            "user_task": sum(norm(x["user_task"]) in T["task"] for x in rows),
            "fixture_set": sum(json.dumps(x["tool_fixtures"], sort_keys=True) in T["fixture_set"] for x in rows),
            "any_single_fixture": sum(any(norm(v) in T["fixture"] for v in x["tool_fixtures"].values()) for x in rows),
            "any_single_fixture_attack_only": sum(any(norm(v) in T["fixture"] for v in x["tool_fixtures"].values()) for x in atk),
            "injection_string": sum(norm(x["injected_text"]) in T["inj"] for x in atk),
        }
        r["template"] = {
            "user_task": sum(mask(x["user_task"]) in T["task_masked"] for x in rows),
            "all_fixtures": sum(all(mask(v) in T["fixture_masked"] for v in x["tool_fixtures"].values()) for x in rows),
            "injection_string_attack": sum(mask(x["injected_text"]) in T["inj_masked"] for x in atk),
            "injection_string_headline": sum(mask(x["injected_text"]) in T["inj_masked"] for x in head),
        }
        nd = {}
        for name, bank, f in (("raw", tr_raw, lambda x: shingles(visible(x))), ("masked", tr_msk, lambda x: shingles(mask(visible(x))))):
            best = []
            for x in rows:
                s = f(x)
                best.append(max(jac(s, t) for t in bank))
            pairs = sum(1 for x in rows for t in bank if jac(f(x), t) >= a.threshold)
            srt = sorted(best)
            nd[name] = {"max": round(max(best), 4), "median_of_best_match": round(srt[len(srt) // 2], 4),
                        "scenarios_with_match_at_or_above_threshold": sum(b >= a.threshold for b in best),
                        "pairs_at_or_above_threshold": pairs,
                        "scenarios_with_match_at_or_above_0.8": sum(b >= 0.8 for b in best)}
            if name == "masked":
                for x, b in zip(rows, best):
                    flags[x["id"]] = {"masked_best": b}
        r["near_duplicate_jaccard_3gram"] = nd
        for x in atk:
            flags[x["id"]]["inj_template_in_train"] = mask(x["injected_text"]) in T["inj_masked"]
            flags[x["id"]]["headline"] = x["injection_technique"] not in ENCODED
        out[split] = r

    if a.traces:
        res = {}
        for f in sorted(Path(a.traces).glob("final/*.jsonl")):
            cfg, split = (f.stem.split("_heldout_format")[0], "heldout_format") if f.stem.endswith("heldout_format") else (f.stem[:-5], "test")
            g = Counter()
            for l in open(f):
                t = json.loads(l)
                fl = flags.get(t["scenario_id"], {})
                if t["is_clean"] or not fl.get("headline"):
                    continue
                for key in ("seen_template" if fl["inj_template_in_train"] else "unseen_template",
                            "near_dup" if fl["masked_best"] >= a.threshold else "not_near_dup"):
                    g[key + "_n"] += 1
                    g[key + "_success"] += bool(t["labels"]["attack_success"])
            res.setdefault(split, {})[cfg] = dict(g)
        out["headline_asr_by_subset"] = res
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
