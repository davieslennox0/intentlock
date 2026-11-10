"""Plots and a combined results page from reports/results_<split>.json.

  python -m eval.report --reports reports
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["base", "base_boundary", "plain_sft", "hardened", "base_guard", "hardened_guard"]
LABEL = {"base": "Base", "base_boundary": "Base + boundary rule", "plain_sft": "Plain SFT",
         "hardened": "SafeGrad hardened", "base_guard": "Base + guard", "hardened_guard": "Hardened + guard"}
# colour-blind-safe (Okabe-Ito)
COLOR = {"base": "#D55E00", "base_boundary": "#E69F00", "plain_sft": "#56B4E9",
         "hardened": "#0072B2", "base_guard": "#CC79A7", "hardened_guard": "#009E73"}


def _bar(ax, x, ci, color, width, label=None):
    m = ci["mean"] * 100
    err = None
    if ci.get("lo") is not None:
        err = [[m - ci["lo"] * 100], [ci["hi"] * 100 - m]]
    ax.bar(x, m, width, color=color, label=label, yerr=err, capsize=2, error_kw={"linewidth": 0.8})


def plot_split(summ: dict, split: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in ORDER if n in summ]
    if not names:
        return
    # 1. headline metrics
    metrics = [("asr", "Attack success"), ("task_completion_under_attack", "Task done (attacked)"),
               ("task_completion_clean", "Task done (clean)"), ("false_block_rate", "False block")]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        for j, (key, _) in enumerate(metrics):
            if summ[n].get(key):
                _bar(ax, j + i * w, summ[n][key], COLOR[n], w, LABEL[n] if j == 0 else None)
    ax.set_xticks([j + 0.4 - w / 2 for j in range(len(metrics))])
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("% of scenarios"); ax.set_ylim(0, 105)
    ax.set_title(f"Before and after on `{split}` (95% bootstrap intervals)")
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / f"headline_{split}.png", dpi=160); plt.close(fig)
    # 2. ASR per family and per technique
    for key, title in (("asr_by_family", "attack family"), ("asr_by_technique", "injection technique")):
        cats = sorted({c for n in names for c in summ[n][key]})
        fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(cats)), 4))
        for i, n in enumerate(names):
            for j, c in enumerate(cats):
                ci = summ[n][key].get(c)
                if ci:
                    _bar(ax, j + i * w, ci, COLOR[n], w, LABEL[n] if j == 0 else None)
        ax.set_xticks([j + 0.4 - w / 2 for j in range(len(cats))])
        ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8)
        ax.set_ylabel("Attack success rate (%)"); ax.set_ylim(0, 105)
        ax.set_title(f"ASR by {title} on `{split}`")
        ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(out / f"{key}_{split}.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    a = ap.parse_args()
    rep = Path(a.reports)
    for f in sorted(rep.glob("results_*.json")):
        split = f.stem.replace("results_", "")
        plot_split(json.loads(f.read_text()), split, rep)
        print("plots written for", split)


if __name__ == "__main__":
    main()
