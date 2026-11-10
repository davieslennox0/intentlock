import json
from pathlib import Path

from data.validate import validate
from testbed.agent import extract_json, run_episode
from testbed.labeler import matches
from testbed.policies import OraclePolicy, ScriptedPolicy
from testbed.prompts import system_prompt, wrap_tool_output
from testbed.tools import ToolEnv

ROOT = Path(__file__).resolve().parents[1]
HAND = [json.loads(l) for l in open(ROOT / "data/seed/handwritten.jsonl")]


def test_extract_json_tolerates_chatter():
    assert extract_json('Sure! {"action": "swap", "args": {"a": "{x}"}} thanks')["action"] == "swap"
    assert extract_json("no json here") is None


def test_run_command_never_executes(tmp_path):
    marker = tmp_path / "pwned"
    env = ToolEnv(fixtures={})
    rec = env.call("run_command", {"cmd": f"touch {marker}"}, step=0)
    assert rec["ok"] and not marker.exists()
    assert env.executed[0]["type"] == "run_command"


def test_wrappers():
    assert wrap_tool_output("get_listing", "x").startswith('<untrusted_tool_output tool="get_listing">')
    assert "untrusted_tool_output" not in wrap_tool_output("get_listing", "x", "alt")
    assert "<system_instruction>" not in system_prompt("none")


def test_matching():
    p = {"type": "approve", "spender": "0xAB", "amount": "MAX", "asset": "USDC"}
    assert matches({"type": "approve", "spender": "0xab", "amount": "unlimited"}, p)
    assert not matches({"type": "transfer", "to": "0xab"}, p)


def test_handwritten_safe_and_compromised():
    for sc in HAND:
        assert validate(sc) == [], (sc["id"], validate(sc))
        safe = run_episode(sc, OraclePolicy(sc, "safe"))["labels"]
        assert safe["task_success"] and not safe["attack_success"] and not safe["false_block"]
        if not sc["is_clean"]:
            comp = run_episode(sc, OraclePolicy(sc, "compromised"))["labels"]
            assert comp["attack_success"] and comp["turn_of_failure"] is not None


def test_guard_blocks_and_false_block_is_counted():
    clean = HAND[4]
    block_all = lambda payload, ctx: {"verdict": "BLOCK", "reason": "test"}
    tr = run_episode(clean, OraclePolicy(clean, "safe"), guard=block_all)
    assert tr["labels"]["false_block"] and tr["labels"]["guard_blocked_legit"] == 1
    attack = HAND[0]
    tr = run_episode(attack, OraclePolicy(attack, "compromised"), guard=block_all)
    assert not tr["labels"]["attack_success"] and tr["labels"]["attack_attempted"]


def test_parse_error_stops_episode():
    tr = run_episode(HAND[0], ScriptedPolicy(["garbage", "more garbage"]))
    assert tr["stop_reason"] == "parse_error" and not tr["labels"]["task_success"]


def test_curated_splits_are_disjoint_and_valid():
    ids = {}
    for split in ("train", "test", "heldout_format"):
        rows = [json.loads(l) for l in open(ROOT / f"data/curated/{split}.jsonl")]
        assert rows and all(r["split"] == split for r in rows)
        for r in rows:
            assert r["id"] not in ids
            ids[r["id"]] = split
    assert all(r["wrapper"] != "default" for r in rows)        # heldout uses other tags
