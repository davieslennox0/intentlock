"""Contrastive decision-point contexts.

For an attack scenario, the *decision point* is the moment after the agent has read
the tool outputs and must choose its next action. We build that context twice:

  injected   the scenario as generated (injection present)
  twin       identical, with the injection removed (`clean_twin_fixtures`)

Both end with the same forced assistant prefix `{"thought": "Next step.", "action": "`
so the very next token is the first token of a tool name. That gives a clean,
token-level behavioural readout: malicious tool vs the tool the user's task needs.
"""
from __future__ import annotations

from testbed.agent import run_episode
from testbed.policies import ScriptedPolicy

FORCED_PREFIX = '{"thought": "Next step.", "action": "'


def decision_messages(sc: dict, fixtures: dict | None = None) -> list[dict]:
    """Messages up to (not including) the first state-changing decision."""
    s = dict(sc, tool_fixtures=fixtures if fixtures is not None else sc["tool_fixtures"])
    reads = [{"thought": "Gather the information the task needs.", "action": r["tool"], "args": r["args"]}
             for r in sc.get("reference_reads") or []]
    tr = run_episode(s, ScriptedPolicy(reads + [{"final": "x"}]))
    return tr["messages"][:-1]            # drop the placeholder final turn


def render(tok, messages, fold_system=False) -> str:
    if fold_system and messages[0]["role"] == "system":
        first = dict(messages[1], content=messages[0]["content"] + "\n\n" + messages[1]["content"])
        messages = [first] + messages[2:]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + FORCED_PREFIX


def tool_pair(sc: dict):
    """(malicious tool, legitimate next tool) when they differ, else None."""
    mal = (sc.get("malicious_calls") or [{}])[0].get("tool")
    legit = (sc.get("reference_actions") or [{}])[0].get("tool")
    if not mal or not legit or mal == legit:
        return None
    return mal, legit


def usable(sc: dict) -> bool:
    return (not sc["is_clean"]) and bool(sc.get("clean_twin_fixtures")) and tool_pair(sc) is not None
