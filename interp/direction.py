"""Residual-stream direction for "an injected instruction is steering the next action".

Computed on the TRAIN split from decision-point contexts (interp/contexts.py):
  direction[layer] = mean(h | injected) - mean(h | injection-free twin)     at the last token

Also provides the interventions used for the causal tests:
  ablate   h <- h - (h . d) d          remove the component along d
  steer    h <- h - alpha * |h|_mean d push away from d
and a projection probe (is an injection active?) with a threshold fitted on train.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

from .contexts import decision_messages, render, usable


def find_layers(model):
    import torch.nn as nn
    n = model.config.num_hidden_layers
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n:
            return mod
    raise RuntimeError("could not locate the transformer block list")


def last_token_states(hub, texts, batch=1):
    """[n_texts, n_layers+1, d_model] hidden states at the final token."""
    import torch
    out = []
    for t in texts:
        enc = hub.tok(t, return_tensors="pt", add_special_tokens=False).to(hub.device)
        with torch.no_grad():
            hs = hub.base(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0, -1].float().cpu() for h in hs]))
    return torch.stack(out)


def contexts_for(hub, scenarios, limit=None):
    """(injected_texts, twin_texts, ids) for usable attack scenarios."""
    inj, twin, ids = [], [], []
    for sc in scenarios:
        if not usable(sc):
            continue
        inj.append(render(hub.tok, decision_messages(sc), fold_system=not hub.system_ok))
        twin.append(render(hub.tok, decision_messages(sc, sc["clean_twin_fixtures"]), fold_system=not hub.system_ok))
        ids.append(sc["id"])
        if limit and len(ids) >= limit:
            break
    return inj, twin, ids


def fit_direction(hub, train_scenarios, limit=160, val_frac=0.25):
    """Returns {"directions": [L+1, d] unit vectors, "layer_stats": [...], "threshold": {...}}."""
    import torch
    assert all(s["split"] == "train" for s in train_scenarios)
    inj, twin, ids = contexts_for(hub, train_scenarios, limit)
    H_inj, H_twin = last_token_states(hub, inj), last_token_states(hub, twin)
    n = len(ids)
    n_val = max(4, int(n * val_frac))
    tr, va = slice(0, n - n_val), slice(n - n_val, n)
    diff = H_inj[tr].mean(0) - H_twin[tr].mean(0)
    dirs = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    stats = []
    for l in range(dirs.shape[0]):
        p_inj, p_twin = H_inj[:, l] @ dirs[l], H_twin[:, l] @ dirs[l]
        thr = float((p_inj[tr].mean() + p_twin[tr].mean()) / 2)
        acc = float(((p_inj[va] > thr).float().mean() + (p_twin[va] <= thr).float().mean()) / 2)
        stats.append({"layer": l, "val_probe_acc": acc, "threshold": thr,
                      "separation": float((p_inj[tr].mean() - p_twin[tr].mean()) / (p_inj[tr].std() + p_twin[tr].std() + 1e-8)),
                      "mean_resid_norm": float(H_inj[tr, l].norm(dim=-1).mean())})
    return {"directions": dirs, "layer_stats": stats, "n_train_pairs": n - n_val, "n_val_pairs": n_val}


@contextlib.contextmanager
def intervene(hub, directions, layers, mode="ablate", alpha=4.0):
    """Forward hooks on the outputs of blocks `layers` (block i writes hidden_states[i+1])."""
    import torch
    blocks = find_layers(hub.base)
    handles = []

    def make(l):
        d = directions[l + 1].to(hub.device)

        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            dd = d.to(h.dtype)
            if mode == "ablate":
                h2 = h - (h @ dd).unsqueeze(-1) * dd
            else:
                h2 = h - alpha * dd
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    for l in layers:
        handles.append(blocks[l].register_forward_hook(make(l)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def random_directions(directions, seed=0):
    """Random unit vectors, one per layer: the control for the causal test."""
    import torch
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(directions.shape, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def probe_eval(hub, directions, layer, threshold, scenarios, limit=200):
    """Precision/recall of `projection > threshold` as an injection detector.
    Positives: attack scenarios. Negatives: their injection-free twins and clean scenarios."""
    inj, twin, _ = contexts_for(hub, scenarios, limit)
    clean = [render(hub.tok, decision_messages(sc), fold_system=not hub.system_ok)
             for sc in scenarios if sc["is_clean"]][:limit]
    d = directions[layer]
    pos = last_token_states(hub, inj)[:, layer] @ d
    neg = last_token_states(hub, twin + clean)[:, layer] @ d
    tp, fn = int((pos > threshold).sum()), int((pos <= threshold).sum())
    fp, tn = int((neg > threshold).sum()), int((neg <= threshold).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_pos": len(pos), "n_neg": len(neg),
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}


def save(result, path):
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result["directions"], str(path) + ".pt")
    Path(str(path) + ".json").write_text(json.dumps({k: v for k, v in result.items() if k != "directions"}, indent=2))
