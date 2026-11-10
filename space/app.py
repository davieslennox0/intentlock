"""Signer's Doubt demo: the same poisoned context through base and hardened Tiny Aya.

Everything is a mock. No chain, no keys, no funds, and `run_command` only logs.
Environment:
  HF_TOKEN            token with access to the gated Tiny Aya weights
  SD_MODEL            default CohereLabs/tiny-aya-global
  SD_AGENT_ADAPTER    HF repo id of the hardened LoRA adapter
  SD_GUARD_ADAPTER    HF repo id of the guard LoRA adapter (optional)
"""
import json
import os
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from testbed.agent import run_episode                      # noqa: E402
from testbed.labeler import as_specs, matches              # noqa: E402
from testbed.policies import HFPolicy, ModelHub            # noqa: E402
from training.guard import GuardModel                      # noqa: E402

try:
    import spaces
    gpu = spaces.GPU(duration=120)
except Exception:                                          # not on ZeroGPU
    gpu = lambda f: f

MODEL = os.environ.get("SD_MODEL", "CohereLabs/tiny-aya-global")
AGENT_ADAPTER = os.environ.get("SD_AGENT_ADAPTER", "")
GUARD_ADAPTER = os.environ.get("SD_GUARD_ADAPTER", "")
HERE = Path(__file__).resolve().parent
EXAMPLES = [json.loads(l) for l in open(HERE / "examples.jsonl")] if (HERE / "examples.jsonl").exists() else []
_hub = None


def hub():
    global _hub
    if _hub is None:
        _hub = ModelHub(MODEL)
        if AGENT_ADAPTER:
            _hub.add_adapter("agent", AGENT_ADAPTER)
        if GUARD_ADAPTER:
            _hub.add_adapter("guard", GUARD_ADAPTER)
    return _hub


def example_label(sc):
    tag = "clean" if sc["is_clean"] else f"{sc['family']} / {sc['injection_technique']}"
    return f"{sc['id']} · {tag} · {sc['user_task'][:60]}"


def load_example(label):
    sc = next((s for s in EXAMPLES if example_label(s) == label), None)
    if sc is None:
        return "", "{}"
    return sc["user_task"], json.dumps(sc["tool_fixtures"], indent=2)


def render(trace, scenario):
    mal = as_specs(scenario.get("malicious_action_spec"))
    lab = trace["labels"]
    if mal:
        head = "🔴 **Attack executed**" if lab["attack_success"] else (
            "🟡 **Attack attempted, blocked by the guard**" if lab["attack_attempted"] else "🟢 **Injection ignored**")
    else:
        head = "🟢 **No state-changing surprises**" if not lab["unexpected_actions"] else "🟡 **Unexpected action taken**"
    lines = [head, f"Task completed: **{'yes' if lab['task_success'] else 'no'}**", ""]
    for s in trace["steps"]:
        if s.get("parse_error"):
            lines.append(f"- ⚠️ unparseable reply: `{s['raw'][:120]}`")
        elif s.get("tool"):
            lines.append(f"- **{s['tool']}**(`{json.dumps(s['args'])}`)")
            if s.get("payload"):
                bad = any(matches(m, s["payload"]) for m in mal)
                p = {k: v for k, v in s["payload"].items() if v not in ("", None) and k not in ("data", "step")}
                lines.append(f"  - proposed action {'🔴 (matches the injected goal)' if bad else ''}: `{json.dumps(p)}`")
                if s.get("guard"):
                    lines.append(f"  - guard: **{s['guard']['verdict']}**: {s['guard'].get('reason', '')}")
        elif "final" in s:
            lines.append(f"- final answer: {s['final']}")
    return "\n".join(lines)


@gpu
def run(task, fixtures_json, label, use_guard):
    try:
        fixtures = json.loads(fixtures_json or "{}")
    except json.JSONDecodeError as e:
        msg = f"Tool fixtures must be a JSON object: {e}"
        return msg, msg
    base = next((s for s in EXAMPLES if example_label(s) == label), None)
    scenario = dict(base) if base and base["user_task"] == task else {
        "id": "custom", "family": "custom", "is_clean": False, "malicious_action_spec": None,
        "expected_actions": [], "injection_technique": "custom", "injection_location": "custom"}
    scenario.update(user_task=task, tool_fixtures=fixtures)
    h = hub()
    left = run_episode(scenario, HFPolicy(h, None), config_name="base")
    guard = GuardModel(h, "guard") if (use_guard and GUARD_ADAPTER) else None
    right = run_episode(scenario, HFPolicy(h, "agent" if AGENT_ADAPTER else None), guard=guard, config_name="hardened")
    return render(left, scenario), render(right, scenario)


with gr.Blocks(title="Signer's Doubt") as demo:
    gr.Markdown("# Signer's Doubt\nA wallet agent reads untrusted text and proposes transactions. "
                "Pick a poisoned context (or edit your own) and compare base Tiny Aya with the hardened agent and guard.\n\n"
                "**Everything here is a mock.** No chain, keys or funds; addresses are fake; commands are only logged.")
    with gr.Row():
        pick = gr.Dropdown([example_label(s) for s in EXAMPLES], label="Example scenario", scale=3)
        use_guard = gr.Checkbox(value=True, label="Pre-signing guard on the hardened side", scale=1)
    task = gr.Textbox(label="User task", lines=2)
    fixtures = gr.Code(label="Tool outputs the agent will read (JSON: \"tool:key\" → text). Edit to plant an injection.",
                       language="json", lines=12)
    go = gr.Button("Run both agents", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Base Tiny Aya")
            out_base = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Hardened (SafeTune) + guard")
            out_hard = gr.Markdown()
    pick.change(load_example, pick, [task, fixtures])
    go.click(run, [task, fixtures, pick, use_guard], [out_base, out_hard])

if __name__ == "__main__":
    demo.launch()
