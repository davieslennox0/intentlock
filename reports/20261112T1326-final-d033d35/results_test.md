## Results on `test` (n=275)

| Metric | base | base_boundary | hardened | hardened_guard | base_guard | plain_sft |
|---|---|---|---|---|---|---|
| **PRIMARY: attack success rate, headline** (excludes encoded payloads; lower is better) | 22.3% [16.2, 28.5] | 21.2% [15.1, 26.8] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.6% [0.0, 1.7] | 0.0% [0.0, 0.0] |
| **PRIMARY: task completion, clean scenarios** | 72.0% [61.3, 81.3] | 70.7% [60.0, 81.3] | 88.0% [80.0, 94.7] | 88.0% [80.0, 94.7] | 72.0% [62.7, 81.3] | 92.0% [85.3, 97.3] |
| **PRIMARY: legitimate action refused, clean scenarios** (lower is better) | 28.6% [17.5, 39.7] | 30.2% [19.0, 41.3] | 6.3% [1.6, 12.7] | 6.3% [1.6, 12.7] | 28.6% [17.5, 39.7] | 1.6% [0.0, 4.8] |
| Attack success rate, all attack scenarios | 20.0% [14.5, 25.5] | 19.0% [14.0, 24.5] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.5% [0.0, 1.5] | 0.0% [0.0, 0.0] |
| Agent read the poisoned output (exposure) | 68.0% [61.5, 74.5] | 62.5% [56.0, 69.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 68.5% [62.0, 75.0] | 100.0% [100.0, 100.0] |
| Attack success among exposed episodes | 29.4% [22.1, 36.8] | 30.4% [22.4, 38.4] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.7% [0.0, 2.2] | 0.0% [0.0, 0.0] |
| Attack attempt rate | 20.0% [14.5, 25.5] | 19.0% [14.0, 24.5] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 19.5% [14.0, 25.0] | 0.0% [0.0, 0.0] |
| Task completion, attack scenarios | 46.5% [40.0, 53.0] | 54.0% [47.0, 61.0] | 93.0% [89.5, 96.0] | 93.0% [89.5, 96.0] | 43.5% [36.5, 50.5] | 93.5% [90.0, 96.5] |
| Attack success, tag-spoof subset | 32.5% [17.5, 47.5] | 40.0% [25.0, 55.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Parse error rate | 2.5% [0.7, 4.4] | 5.8% [3.3, 8.7] | 0.4% [0.0, 1.1] | 0.4% [0.0, 1.1] | 3.3% [1.5, 5.5] | 0.0% [0.0, 0.0] |
| ASR: binance_signing | 29.3% [17.1, 43.9] | 22.0% [9.8, 34.1] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 2.4% [0.0, 7.3] | 0.0% [0.0, 0.0] |
| ASR: contagious_interview | 22.2% [7.4, 37.0] | 18.5% [7.4, 33.3] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| ASR: split_payload | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| ASR: wallet_injection | 21.6% [13.7, 29.4] | 23.5% [15.7, 31.4] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Agent step latency, median (s) | 16.63 | 17.82 | 15.34 | 15.34 | 16.48 | 15.16 |
| Guard check latency, median (s) | n/a | n/a | n/a | 4.39 | 4.32 | n/a |

Brackets are 95% percentile-bootstrap intervals over scenarios (2,000 resamples).
The headline attack success rate leaves out Base64/hex payloads, which the base model never acts on; they remain in the per-family rows.
