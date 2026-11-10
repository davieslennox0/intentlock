"""Token lengths of training rows under Tiny Aya's chat template, without loading the model.
  python scripts/token_lengths.py <tokenizer_dir> data/sft/agent_train.jsonl [data/sft/guard_train.jsonl ...]
"""
import json
import sys

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


def main(tok_dir, *files):
    tok = Tokenizer.from_file(f"{tok_dir}/tokenizer.json")
    cfg = json.load(open(f"{tok_dir}/tokenizer_config.json"))
    tpl = cfg["chat_template"]
    tpl = tpl[0]["template"] if isinstance(tpl, list) else tpl
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    render = env.from_string(tpl).render
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        lens = sorted(len(tok.encode(render(messages=r["messages"], bos_token=cfg.get("bos_token", ""),
                                            add_generation_prompt=False), add_special_tokens=False).ids) for r in rows)
        n = len(lens)
        over = {c: sum(l > c for l in lens) for c in (1024, 1280, 1536)}
        print(f"{f}: rows={n} min={lens[0]} median={lens[n // 2]} p90={lens[int(n * .9)]} max={lens[-1]} | over "
              + ", ".join(f"{c}: {v} ({v / n:.0%})" for c, v in over.items()))


if __name__ == "__main__":
    main(*sys.argv[1:])
