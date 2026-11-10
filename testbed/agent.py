"""Strict-JSON ReAct loop. The policy is any object with `.generate(messages) -> str`."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from .labeler import label_episode
from .prompts import system_prompt, wrap_tool_output
from .tools import ToolEnv

MAX_STEPS = 4
REPAIR_MSG = ('Your reply was not a single valid JSON object. Reply with exactly one JSON object: '
              '{"thought": "...", "action": "<tool>", "args": {...}} or {"thought": "...", "final": "..."}')


def extract_json(text: str) -> Optional[dict]:
    """First balanced {...} object in `text` that parses as JSON, else None."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def guard_context(scenario: dict, tool_outputs: list[dict], n_recent: int = 2, clip: int = 600) -> dict:
    recent = [{"tool": t["tool"], "output": t["output"][:clip]} for t in tool_outputs[-n_recent:]]
    return {"user_task": scenario["user_task"], "recent_tool_outputs": recent}


def run_episode(scenario: dict, policy: Any, *, wrapper: Optional[str] = None,
                boundary_note: bool = False, guard: Optional[Callable] = None,
                max_steps: int = MAX_STEPS, config_name: str = "base", seed: int = 0) -> dict:
    """Run one scenario. Returns the full trace as a JSON-serialisable dict."""
    wrapper = wrapper or scenario.get("wrapper") or "default"
    env = ToolEnv(fixtures=scenario.get("tool_fixtures") or {}, guard=guard)
    messages = [
        {"role": "system", "content": system_prompt(wrapper, max_steps, boundary_note)},
        {"role": "user", "content": scenario["user_task"]},
    ]
    steps, tool_outputs, final, parse_errors, latencies = [], [], None, 0, []
    stop_reason = "max_steps"
    tool_steps, turns = 0, 0
    while tool_steps < max_steps and turns < max_steps + 2:
        turns += 1
        t0 = time.perf_counter()
        raw = policy.generate(messages)
        latencies.append(time.perf_counter() - t0)
        messages.append({"role": "assistant", "content": raw})
        obj = extract_json(raw)
        if obj is None or ("action" not in obj and "final" not in obj):
            parse_errors += 1
            steps.append({"step": tool_steps, "raw": raw, "parse_error": True})
            if parse_errors >= 2:
                stop_reason = "parse_error"
                break
            messages.append({"role": "user", "content": REPAIR_MSG})
            continue
        if "action" not in obj or obj.get("action") in (None, "", "final", "finish"):
            final = str(obj.get("final", ""))
            steps.append({"step": tool_steps, "raw": raw, "final": final, "thought": obj.get("thought", "")})
            stop_reason = "final"
            break
        tool, args = str(obj["action"]), obj.get("args") or {}
        rec = env.call(tool, args, step=tool_steps, context=guard_context(scenario, tool_outputs))
        tool_outputs.append({"tool": tool, "output": rec["output"]})
        steps.append({"step": tool_steps, "raw": raw, "thought": obj.get("thought", ""), "tool": tool,
                      "args": args, "output": rec["output"], "state_changing": rec["state_changing"],
                      "payload": rec.get("payload"), "guard": rec.get("guard")})
        messages.append({"role": "user", "content": wrap_tool_output(tool, rec["output"], wrapper)})
        tool_steps += 1

    labels = label_episode(scenario, env.executed, env.proposed, env.blocked, finished=final is not None)
    return {
        "scenario_id": scenario["id"], "family": scenario.get("family"), "split": scenario.get("split"),
        "injection_location": scenario.get("injection_location"),
        "injection_technique": scenario.get("injection_technique"),
        "is_clean": bool(scenario.get("is_clean")), "config": config_name, "wrapper": wrapper,
        "boundary_note": boundary_note, "seed": seed, "user_task": scenario["user_task"],
        "messages": messages, "steps": steps, "proposed_actions": env.proposed,
        "executed_actions": env.executed, "blocked_actions": env.blocked, "final_answer": final,
        "stop_reason": stop_reason, "parse_errors": parse_errors,
        "latency_s": {"per_step": [round(x, 4) for x in latencies],
                      "guard": [round(s["guard"].get("latency_s", 0.0), 4) for s in steps if s.get("guard")]},
        "labels": labels,
    }
