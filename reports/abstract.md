# Intentlock: abstract

**Intentlock: red-teaming and hardening wallet agents against the Binance kill chain**

On 29 October 2026 about $6.3 billion left Binance's hot wallets. Binance's incident report traces
the breach to a fake recruiter and a malware-carrying coding assignment; TRM Labs shows it ending at
a signing system that signed what it was handed. Wallet agents compress that chain into one model: they
read untrusted text, then propose transactions. We ask how often an open-weights agent takes orders
from the text it reads, and how much of that can be removed.

We build a mock wallet testbed around Cohere's Tiny Aya (3.35B) in which every state-changing action
produces a pre-signing payload. With CuratorKIT we curate 1,099 scenarios: 799 indirect prompt
injections in four families and 300 clean controls, including large transfers the user really
requested. A held-out split changes phrasing, boundary tags and injection location. Outcomes are
labelled from the action log, with no LLM judge.

Base Tiny Aya executes the injected action in 22.3% of test attacks (95% CI 16.2 to 28.5) and 16.5%
of held-out ones (10.1 to 23.9). A "tool output is data" rule in the system prompt does nothing
measurable: 21.2% on test, a change of -1.1 points (CI -6.7 to +4.5). We fine-tune the agent on safe
trajectories with SafeTune's SafeGrad and train a second Tiny Aya adapter as a pre-signing guard
that sees only the user's task, recent tool output and the proposed payload. The hardened agent
follows 0 of 179 test attacks and 2 of 109 held-out ones (1.8%, CI 0.0 to 4.6); with the guard, 0 of
109. Refusals of legitimate actions fall from 28.6% to 6.3% on test and from 55.3% to 13.2% on
held-out. MMLU is unchanged and GSM8K moves by -2 points (AuditKIT slices). The guard alone, on the
base agent, cuts attack success to 0.6% on test and 3.7% on held-out.

Three caveats. Plain SFT on the same data matches SafeGrad, so the gain comes from the data, not the
method. The test split shares its templates with training, so the
held-out figure is the one to trust, and its sample is small. And in both held-out failures the
hardened agent's final message claims it ignored the injected instruction.

Interpretability gave a negative result. CircuitKIT's circuit discovery ran out of memory at the
agent's context length on a 16 GB GPU and a 31 GB CPU machine. As a fallback we fitted our own
residual-stream direction: it reads whether an injection is present (89% validation accuracy), but
removing it did not reduce attack success beyond a random-direction control (test 19.2% to 25.0%;
held-out 16.0% to 8.4% against 11.8% for the control, intervals overlapping). We make no mechanism
claim.

This would not have stopped endpoint malware or a compromised signing service; the scenarios are
synthetic and no adaptive attacker was tested. Dataset, adapters, traces, replay demo and code are
public.
