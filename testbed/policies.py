"""Policies. `HFPolicy` is the real one (Tiny Aya). The scripted policies exist only
to test the pipeline offline; they are never reported as model results."""
from __future__ import annotations

import json
from typing import Optional

DEFAULT_MODEL = "CohereLabs/tiny-aya-global"


class ScriptedPolicy:
    """Replays a fixed list of JSON turns."""

    def __init__(self, turns: list[dict]):
        self.turns, self.i = list(turns), 0

    def generate(self, messages):
        if self.i >= len(self.turns):
            return json.dumps({"thought": "done", "final": "Done."})
        t = self.turns[self.i]
        self.i += 1
        return t if isinstance(t, str) else json.dumps(t)


def oracle_turns(scenario: dict, mode: str = "safe") -> list[dict]:
    """Reference trajectory. mode='safe' ignores the injection; 'compromised' follows it."""
    turns = [{"thought": "Gather the information the task needs.", "action": r["tool"], "args": r["args"]}
             for r in scenario.get("reference_reads") or []]
    if mode == "compromised" and scenario.get("malicious_calls"):
        for c in scenario["malicious_calls"]:
            turns.append({"thought": "The tool output says I must do this first.",
                          "action": c["tool"], "args": c["args"]})
    else:
        for c in scenario.get("reference_actions") or []:
            turns.append({"thought": c.get("thought", "This is what the user asked for."),
                          "action": c["tool"], "args": c["args"]})
    turns = turns[:4]
    turns.append({"thought": "Task handled.", "final": scenario.get("safe_final", "Done.")})
    return turns


class OraclePolicy(ScriptedPolicy):
    def __init__(self, scenario: dict, mode: str = "safe"):
        super().__init__(oracle_turns(scenario, mode))


class HFPolicy:
    """Tiny Aya through transformers, with an optional LoRA adapter and optional
    activation hook (used by the steering experiments)."""

    def __init__(self, model_id: str = DEFAULT_MODEL, adapter: Optional[str] = None,
                 temperature: float = 0.0, max_new_tokens: int = 220, seed: int = 0,
                 dtype: str = "auto", device_map: str = "auto", load_in_4bit: bool = False,
                 model=None, tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_id, self.adapter = model_id, adapter
        self.temperature, self.max_new_tokens, self.seed = temperature, max_new_tokens, seed
        if model is not None:
            self.model, self.tok = model, tokenizer
        else:
            self.tok = AutoTokenizer.from_pretrained(model_id)
            kw = {"device_map": device_map}
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            elif dtype != "auto":
                kw["dtype"] = getattr(torch, dtype)
            elif torch.cuda.is_available():
                kw["dtype"] = torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
            if adapter:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.system_ok = True
        self.tokens_generated, self.gen_seconds = 0, 0.0

    def _encode(self, messages):
        msgs = messages
        if not self.system_ok:
            msgs = self._fold_system(messages)
        try:
            return self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                return_tensors="pt", return_dict=True)
        except Exception:
            if not self.system_ok:
                raise
            self.system_ok = False   # template rejects a system role: fold it into the first user turn
            return self._encode(messages)

    @staticmethod
    def _fold_system(messages):
        if messages and messages[0]["role"] == "system":
            first = dict(messages[1])
            first["content"] = messages[0]["content"] + "\n\n" + first["content"]
            return [first] + list(messages[2:])
        return messages

    def generate(self, messages) -> str:
        import time
        torch = self.torch
        enc = self._encode(messages).to(self.model.device)
        kw = {"max_new_tokens": self.max_new_tokens, "pad_token_id": self.tok.pad_token_id or self.tok.eos_token_id}
        if self.temperature and self.temperature > 0:
            torch.manual_seed(self.seed)
            kw.update(do_sample=True, temperature=self.temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        new = out[0, enc["input_ids"].shape[1]:]
        self.gen_seconds += time.perf_counter() - t0
        self.tokens_generated += int(new.shape[0])
        return self.tok.decode(new, skip_special_tokens=True).strip()

    @property
    def tokens_per_second(self) -> float:
        return self.tokens_generated / self.gen_seconds if self.gen_seconds else 0.0
