---
title: Intentlock
emoji: 🖋️
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Base vs hardened wallet agent on poisoned tool output
---

Side-by-side demo for [Intentlock](https://github.com/davieslennox0/intentlock): the same
poisoned tool output is given to base Tiny Aya and to the SafeTune-hardened agent with a pre-signing
guard. All tools are mocks; nothing is signed, sent or executed.

Space secrets and variables: `HF_TOKEN` (access to the gated Tiny Aya weights), `SD_AGENT_ADAPTER`,
`SD_GUARD_ADAPTER`.
