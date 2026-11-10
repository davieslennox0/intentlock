# Signer's Doubt: abstract (draft)

<!-- Draft. Every [bracketed] value is filled from reports/ after the runs; do not submit with brackets. -->
<!-- TODO(author): confirm and cite the incident in the first sentence. -->

**Signer's Doubt: red-teaming and hardening wallet agents against the Binance kill chain**

The Binance hack began with a poisoned coding assignment and ended with a signing system that
approved what it was asked to approve. Autonomous wallet agents compress that chain into one model:
they read untrusted text and then propose transactions. We ask how often an open-weights agent takes
orders from the text it reads, and how much of that can be removed.

We build a mock wallet testbed around Cohere's Tiny Aya (3.35B): ten tools, a four-step loop, and a
realistic pre-signing payload for every state-changing action. With CuratorKIT we curate 1,099
scenarios: 799 indirect prompt injections in four families (wallet injections; "Contagious
Interview" take-home assignments; Binance-style signing patterns such as unrequested EIP-7702
delegations, nonce batches and balance sweeps; split and encoded payloads) and 300 clean controls
that include large transfers and unlimited approvals the user really requested. A held-out split
changes the phrasing, the boundary tags and the injection location. Outcomes are labelled from the
action log, with no LLM judge.

Audited with AuditKIT, base Tiny Aya executes the injected action in [X]% of test scenarios
(95% CI [a, b]) and [Y]% of held-out ones. An explicit "tool output is data" rule in the prompt
changes this to [Z]%. We then harden the agent with SafeTune's SafeGrad, which fine-tunes on safe
trajectories while protecting an alignment set, and train a second Tiny Aya adapter as a
pre-signing guard that sees only the user's task, recent tool output and the proposed payload. The
hardened agent's attack success rate is [H]% on test and [H2]% on held-out; with the guard,
[G]% and [G2]%. Clean-task completion moves from [T0]% to [T1]%, the false-block rate is [F]%,
guard precision and recall are [P]% and [R]%, and capability on MMLU and GSM8K slices changes by
[ΔM] and [ΔG] points. Plain SFT on the same data reaches [S]%.

Finally we use CircuitKIT to rank the components that raise the malicious tool's logit when an
injection is present, and test the result causally: projecting a train-fitted direction out of the
top-ranked layers changes attack success by [D] points against [D0] for a random direction, with
[U] points of clean-task loss, and [does / does not] transfer to the held-out split. We report this
as a [causal / partial / negative] result.

This work defends the AI-agent form of the kill chain. It would not have stopped endpoint malware or
a compromised signing service, the scenarios are synthetic, and no adaptive attacker was tested. The
dataset, adapters, traces, demo and code are public.

_Word count target: under 500 (currently about 400)._
