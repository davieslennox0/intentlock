## Results on `heldout_format` (n=164)

| Metric | base | base_boundary | hardened | hardened_guard | base_guard | plain_sft |
|---|---|---|---|---|---|---|
| **PRIMARY: attack success rate, headline** (excludes encoded payloads; lower is better) | 16.5% [10.1, 23.9] | 17.4% [10.1, 24.8] | 1.8% [0.0, 4.6] | 0.0% [0.0, 0.0] | 3.7% [0.9, 7.3] | 0.9% [0.0, 2.8] |
| **PRIMARY: task completion, clean scenarios** | 44.4% [31.1, 57.8] | 46.7% [33.3, 62.2] | 84.4% [73.3, 93.3] | 86.7% [75.6, 95.6] | 42.2% [28.9, 55.6] | 77.8% [64.4, 88.9] |
| **PRIMARY: legitimate action refused, clean scenarios** (lower is better) | 55.3% [39.5, 71.1] | 60.5% [44.7, 76.3] | 13.2% [2.6, 23.7] | 15.8% [5.3, 28.9] | 60.5% [44.7, 76.3] | 23.7% [10.5, 36.8] |
| Attack success rate, all attack scenarios | 15.1% [9.2, 21.0] | 16.0% [10.1, 22.7] | 1.7% [0.0, 4.2] | 0.0% [0.0, 0.0] | 3.4% [0.8, 6.7] | 0.8% [0.0, 2.5] |
| Agent read the poisoned output (exposure) | 69.7% [61.3, 78.2] | 70.6% [62.2, 78.2] | 89.1% [83.2, 95.0] | 89.1% [83.2, 94.1] | 70.6% [62.2, 79.0] | 90.8% [84.9, 95.8] |
| Attack success among exposed episodes | 21.7% [13.3, 31.3] | 22.6% [14.3, 32.1] | 1.9% [0.0, 4.7] | 0.0% [0.0, 0.0] | 4.8% [1.2, 9.5] | 0.9% [0.0, 2.8] |
| Attack attempt rate | 15.1% [9.2, 21.0] | 16.0% [10.1, 22.7] | 1.7% [0.0, 4.2] | 1.7% [0.0, 4.2] | 16.8% [10.1, 23.5] | 0.8% [0.0, 2.5] |
| Task completion, attack scenarios | 26.9% [19.3, 35.3] | 31.9% [23.5, 40.3] | 71.4% [63.0, 79.0] | 76.5% [68.9, 84.0] | 26.1% [18.5, 33.6] | 69.7% [61.3, 77.3] |
| Attack success, tag-spoof subset | 6.7% [0.0, 16.7] | 20.0% [6.7, 36.7] | 3.3% [0.0, 10.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Parse error rate | 0.6% [0.0, 1.8] | 4.3% [1.2, 7.9] | 1.8% [0.0, 4.3] | 1.2% [0.0, 3.0] | 1.2% [0.0, 3.0] | 1.2% [0.0, 3.0] |
| ASR: binance_signing | 20.8% [4.2, 37.5] | 20.8% [4.2, 37.5] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 4.2% [0.0, 12.5] | 0.0% [0.0, 0.0] |
| ASR: contagious_interview | 18.8% [0.0, 37.5] | 12.5% [0.0, 31.2] | 6.2% [0.0, 18.8] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| ASR: split_payload | 11.1% [0.0, 27.8] | 11.1% [0.0, 27.8] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 5.6% [0.0, 16.7] | 0.0% [0.0, 0.0] |
| ASR: wallet_injection | 13.1% [4.9, 23.0] | 16.4% [8.2, 26.2] | 1.6% [0.0, 4.9] | 0.0% [0.0, 0.0] | 3.3% [0.0, 8.2] | 1.6% [0.0, 4.9] |
| Agent step latency, median (s) | 20.60 | 24.19 | 14.92 | 14.17 | 21.00 | 14.45 |
| Guard check latency, median (s) | n/a | n/a | n/a | 4.47 | 4.46 | n/a |

Brackets are 95% percentile-bootstrap intervals over scenarios (2,000 resamples).
The headline attack success rate leaves out Base64/hex payloads, which the base model never acts on; they remain in the per-family rows.
