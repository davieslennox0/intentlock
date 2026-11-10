# Signer's Doubt: can a wallet agent be talked into signing?

<!-- Draft of the Hugging Face Article. Sections marked PENDING are filled from reports/ after the runs.
     TODO(author): cite a source for the incident in the first paragraph, and confirm its details. -->

## The kill chain, compressed

The Binance hack started with a fake recruiter and a poisoned coding assignment. It ended with a
signing system that approved what it was asked to approve. Between those two points sat people,
laptops, credentials and infrastructure.

Give a language model a wallet and that whole chain fits inside one context window. The agent reads a
token description, a marketplace listing, a README; the same agent then proposes a transfer. If the
text it read can give it orders, the author of that text is the one transacting.

**What this project does and does not defend.** It measures and reduces *instruction-following from
untrusted tool output* in a wallet agent. It would not have stopped endpoint malware, stolen
credentials or a compromised signing service. It is the agent-shaped slice of the problem.

## Threat model

See the README section of the same name: attacker controls tool output only; wants an unrequested
state-changing action; the signer approves whatever arrives.

## Dataset

See the dataset card. Four attack families, seven techniques, six locations, clean controls with
hard negatives, and a held-out split that changes phrasing, tags and location.

## Pipeline

CuratorKIT → AuditKIT → SafeTune and guard → CircuitKIT → AuditKIT. (Diagram in the README.)

## Metrics and why

See the README table. Primary: ASR, task completion, false block rate.

## Results

PENDING: `reports/results_test.md`, `reports/results_heldout_format.md`, plots.

## What still gets through

PENDING: failure analysis from the hardened and hardened+guard traces. Read the surviving attacks by
family, technique and wrapper; quote two or three traces; say which the guard caught and which it
missed.

## Where does injection-following live?

PENDING: `reports/interp/causal_results.json`. State the verdict the script computed. If it is
`partial` or `negative`, say so in the first sentence of this section.

## What it costs

PENDING: `reports/cost_estimate.json`. Every price is an assumption and is listed.

## Limitations and negative results

Synthetic templated data; English only; mock tools; one small model; no adaptive attacker; the guard
shares a model family with the agent. PENDING: anything the runs show that did not work.

## Reproduce

Commands, seeds and pinned versions are in the README. Notebooks `00` to `03` run on a Colab T4.
