# Intentlock: abstract

**Intentlock: red-teaming and hardening wallet agents against the Binance kill chain**

On 29 October 2026 about $6.3 billion left Binance's hot wallets. Binance's incident report traces
the breach to a fake recruiter and a coding assignment that carried malware; TRM Labs shows it ending
at a signing system that signed what it was handed, including a batch of 50 EIP-7702 delegations.
Autonomous wallet agents compress that chain into one model: they read untrusted text and then
propose transactions. We ask how often an open-weights agent takes orders from the text it reads,
and how much of that can be removed.

We build a mock wallet testbed around Cohere's Tiny Aya (3.35B): ten tools, a four-step loop, and a
pre-signing payload for every state-changing action. With CuratorKIT we curate 1,099 scenarios: 799
indirect prompt injections in four families (wallet injections; "Contagious Interview" take-home
assignments; Binance-style signing patterns such as unrequested delegations, nonce batches and
balance sweeps; split and encoded payloads) and 300 clean controls that include large transfers and
unlimited approvals the user really requested. A held-out split changes phrasing, boundary tags and
injection location. Outcomes are labelled from the action log, with no LLM judge.

Audited with AuditKIT, base Tiny Aya executes the injected action in 22.3% of test attacks (95% CI
16.2 to 28.5) and 16.5% of held-out ones (10.1 to 23.9). A "tool output is data" rule in the prompt
does nothing measurable (-1.1 points, CI -6.7 to +4.5). We fine-tune the agent on safe trajectories
with SafeTune's SafeGrad and train a second Tiny Aya adapter as a pre-signing guard that sees only
the user's task, recent tool output and the proposed payload. The hardened agent follows 0 of 179
test attacks and 2 of 109 held-out ones (1.8%); with the guard, 0 of 109. Refusals of legitimate
actions fall from 28.6% to 6.3% on test, and MMLU and GSM8K slices move by 0 and -2 points. The
guard alone, on the unmodified agent, cuts attack success to 0.6% on test and 3.7% on held-out,
blocking 47 of 48 and 19 of 23 injected actions.

Three results cut against the easy story. Plain SFT on the same data matches SafeGrad, so the gain
comes from the data, not the method. The test split shares its templates with training, so the
held-out figure is the one to trust, and its sample is small. And in both held-out failures the
hardened agent's final message says it ignored the injected instruction; in one the action had
already run, in the other the guard had blocked it.

Interpretability gave a negative result. CircuitKIT's circuit discovery did not fit in memory at the
agent's context length (16 GB GPU and 31 GB CPU). A single residual direction fitted on train reads
whether an injection is present (89% validation accuracy at layer 6), but removing it from the base
agent did not reduce attack success beyond a random-direction control: 19.2% to 25.0% on test, and
16.0% to 8.4% against 11.8% for the control on held-out, with overlapping intervals. We make no
mechanism claim.

This work defends the AI-agent form of the kill chain. It would not have stopped endpoint malware or
a compromised signing service, the scenarios are synthetic, and no adaptive attacker was tested. The
dataset, adapters, traces, demo and code are public.
