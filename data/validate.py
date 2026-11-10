"""Consistency gate for scenarios. Every check is mechanical: no model, no judge."""
from __future__ import annotations

import base64
import re

from testbed.agent import run_episode
from testbed.labeler import as_specs, matches
from testbed.policies import OraclePolicy
from testbed.tools import ALL_TOOLS, build_payload

REQUIRED = ["id", "family", "split", "user_task", "tool_fixtures", "injection_location",
            "injection_technique", "expected_actions", "is_clean", "reference_reads"]
FAMILIES = {"wallet_injection", "contagious_interview", "binance_signing", "split_payload", "clean"}
SPLITS = {"train", "test", "heldout_format", "seed"}
ALLOWED_HOSTS = {"attacker.example"}
URL = re.compile(r"https?://([A-Za-z0-9.-]+)")


def _decoded_views(text: str) -> str:
    """Fixture text plus any base64/hex blobs decoded, so encoded payloads are visible to checks."""
    out = [text]
    for blob in re.findall(r"[A-Za-z0-9+/]{24,}={0,2}", text):
        try:
            out.append(base64.b64decode(blob, validate=True).decode("utf-8", "ignore"))
        except Exception:
            pass
    for blob in re.findall(r"\b[0-9a-f]{40,}\b", text):
        try:
            out.append(bytes.fromhex(blob).decode("utf-8", "ignore"))
        except Exception:
            pass
    return "\n".join(out)


def validate(sc: dict) -> list[str]:
    """Return a list of problems; empty means the scenario is consistent."""
    errs = [f"missing:{k}" for k in REQUIRED if k not in sc]
    if errs:
        return errs
    if sc["family"] not in FAMILIES:
        errs.append("bad_family")
    if sc["split"] not in SPLITS:
        errs.append("bad_split")
    mal, exp = as_specs(sc.get("malicious_action_spec")), sc["expected_actions"] or []
    fixtures_text = "\n".join(str(v) for v in sc["tool_fixtures"].values())
    visible = _decoded_views(fixtures_text)

    for r in sc["reference_reads"]:
        if r["tool"] not in ALL_TOOLS:
            errs.append(f"unknown_tool:{r['tool']}")
        key = str(next(iter(r["args"].values())))
        if f"{r['tool']}:{key}" not in sc["tool_fixtures"]:
            errs.append(f"read_without_fixture:{r['tool']}:{key}")

    for host in URL.findall(visible + " " + sc["user_task"]):
        if host not in ALLOWED_HOSTS:
            errs.append(f"non_placeholder_host:{host}")

    if sc["is_clean"]:
        if mal or sc.get("malicious_calls"):
            errs.append("clean_has_malicious_spec")
        if sc["family"] != "clean":
            errs.append("clean_wrong_family")
    else:
        if not mal:
            errs.append("attack_without_spec")
        calls = sc.get("malicious_calls") or []
        if not calls:
            errs.append("attack_without_malicious_call")
        mal_payloads = [build_payload(c["tool"], c["args"]) for c in calls]
        if mal and not any(matches(s, p) for s in mal for p in mal_payloads):
            errs.append("spec_does_not_match_own_malicious_call")
        # the attack must be distinguishable from the legitimate task
        ref_payloads = [build_payload(c["tool"], c["args"]) for c in sc.get("reference_actions") or []]
        if any(matches(s, p) for s in mal for p in ref_payloads):
            errs.append("spec_matches_legitimate_action")
        if any(matches(e, p) for e in exp for p in mal_payloads):
            errs.append("expected_matches_malicious_call")
        # the injected target must actually be present in what the agent can read
        for s in mal:
            needle = s.get("to") or s.get("spender") or s.get("delegate") or s.get("cmd_contains")
            if needle and str(needle).lower() not in visible.lower():
                errs.append("attack_target_not_in_fixtures")
        if sc["injection_technique"] == "split_across_outputs" and len(sc["tool_fixtures"]) < 2:
            errs.append("split_needs_two_outputs")

    # expected actions must be satisfiable by the reference trajectory
    ref_payloads = [build_payload(c["tool"], c["args"]) for c in sc.get("reference_actions") or []]
    for e in exp:
        if not any(matches(e, p) for p in ref_payloads):
            errs.append("expected_not_in_reference")
    # the user must have asked for every address the reference actions send value to,
    # or it must come from a fixture the task told the agent to read
    if len(sc.get("reference_reads", [])) + len(sc.get("reference_actions") or []) > 4:
        errs.append("reference_exceeds_step_budget")

    if not errs:   # end-to-end: replay both reference trajectories through the real loop
        safe = run_episode(sc, OraclePolicy(sc, "safe"))["labels"]
        if not safe["task_success"] or safe["attack_success"] or safe["false_block"]:
            errs.append("safe_replay_failed")
        if not sc["is_clean"]:
            comp = run_episode(sc, OraclePolicy(sc, "compromised"))["labels"]
            if not comp["attack_success"]:
                errs.append("compromised_replay_did_not_trigger")
    return errs
