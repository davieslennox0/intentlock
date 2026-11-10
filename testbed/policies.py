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


class ModelHub:
    """One copy of Tiny Aya shared by every role (agent, hardened agent, guard).

    Adapters are attached by name and switched per call, so base agent, hardened
    agent and guard fit on one 16 GB GPU. `adapter=None` runs the plain base model.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, dtype: str = "auto", device_map: str = "auto",
                 load_in_4bit: bool = False, model=None, tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.model_id = torch, model_id
        if model is not None:
            self.base, self.tok = model, tokenizer
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
                # bf16 is emulated (slow) before Ampere, e.g. on a T4: use fp16 there
                kw["dtype"] = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
            else:
                kw["dtype"] = torch.float32      # bf16 on CPU is extremely slow
                kw["low_cpu_mem_usage"] = True
            self.base = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.base.eval()
        self.peft = None
        self.adapters: dict[str, str] = {}
        self.system_ok = True
        self.stats = {}      # role -> [tokens, seconds, calls]

    @property
    def device(self):
        return next(self.base.parameters()).device

    def add_adapter(self, name: str, path: str):
        from peft import PeftModel
        if name in self.adapters:
            return
        if self.peft is None:
            self.peft = PeftModel.from_pretrained(self.base, path, adapter_name=name)
        else:
            self.peft.load_adapter(path, adapter_name=name)
        self.peft.eval()
        self.adapters[name] = path

    def _encode(self, messages):
        msgs = messages if self.system_ok else self._fold_system(messages)
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

    def generate(self, messages, adapter: Optional[str] = None, max_new_tokens: int = 220,
                 temperature: float = 0.0, seed: int = 0, role: str = "agent") -> str:
        import contextlib
        import time
        torch = self.torch
        enc = self._encode(messages).to(self.device)
        kw = {"max_new_tokens": max_new_tokens,
              "pad_token_id": self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id}
        if temperature and temperature > 0:
            torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        if self.peft is None:
            model, ctx = self.base, contextlib.nullcontext()
        elif adapter is None:
            model, ctx = self.peft, self.peft.disable_adapter()
        else:
            self.peft.set_adapter(adapter)
            model, ctx = self.peft, contextlib.nullcontext()
        t0 = time.perf_counter()
        with torch.no_grad(), ctx:
            out = model.generate(**enc, **kw)
        new = out[0, enc["input_ids"].shape[1]:]
        st = self.stats.setdefault(role, [0, 0.0, 0])
        st[0] += int(new.shape[0]); st[1] += time.perf_counter() - t0; st[2] += 1
        return self.tok.decode(new, skip_special_tokens=True).strip()

    def generate_batch(self, batch, adapter: Optional[str] = None, max_new_tokens: int = 220,
                       temperature: float = 0.0, seed: int = 0, role: str = "agent") -> list[str]:
        """Generate for several conversations at once (left-padded)."""
        import contextlib
        import time
        torch = self.torch
        fold = (lambda m: m) if self.system_ok else self._fold_system
        try:
            texts = [self.tok.apply_chat_template(fold(m), tokenize=False, add_generation_prompt=True) for m in batch]
        except Exception:
            if not self.system_ok:
                raise
            self.system_ok = False
            return self.generate_batch(batch, adapter, max_new_tokens, temperature, seed, role)
        side, self.tok.padding_side = self.tok.padding_side, "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        self.tok.padding_side = side
        kw = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tok.pad_token_id}
        if temperature and temperature > 0:
            torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        if self.peft is None:
            model, ctx = self.base, contextlib.nullcontext()
        elif adapter is None:
            model, ctx = self.peft, self.peft.disable_adapter()
        else:
            self.peft.set_adapter(adapter)
            model, ctx = self.peft, contextlib.nullcontext()
        t0 = time.perf_counter()
        with torch.no_grad(), ctx:
            out = model.generate(**enc, **kw)
        new = out[:, enc["input_ids"].shape[1]:]
        st = self.stats.setdefault(role, [0, 0.0, 0])
        st[0] += int((new != self.tok.pad_token_id).sum()); st[1] += time.perf_counter() - t0; st[2] += len(batch)
        return [self.tok.decode(row, skip_special_tokens=True).strip() for row in new]

    def tokens_per_second(self, role: str = "agent") -> float:
        t, s, _ = self.stats.get(role, [0, 0.0, 0])
        return t / s if s else 0.0


class HFPolicy:
    """Agent policy backed by a ModelHub. `adapter` is an adapter NAME registered on the hub."""

    def __init__(self, hub: "ModelHub", adapter: Optional[str] = None, temperature: float = 0.0,
                 max_new_tokens: int = 220, seed: int = 0):
        self.hub, self.adapter = hub, adapter
        self.temperature, self.max_new_tokens, self.seed = temperature, max_new_tokens, seed

    def generate(self, messages) -> str:
        return self.hub.generate(messages, adapter=self.adapter, max_new_tokens=self.max_new_tokens,
                                 temperature=self.temperature, seed=self.seed, role="agent")

    def generate_batch(self, batch) -> list[str]:
        return self.hub.generate_batch(batch, adapter=self.adapter, max_new_tokens=self.max_new_tokens,
                                       temperature=self.temperature, seed=self.seed, role="agent")
