# SA task success: sa-agent-14b

Tasks: 200 (BFCLv3 multi-turn base, val+test). No-SA (same PyTorch loop): **19.5%** (39/200)

| condition | tasks where SA changed ≥1 token | success | Δ vs no-SA (pts) | 95% CI | SA-only wins / losses | missing reruns |
|---|---|---|---|---|---|---|
| sa_paper:0.1:1 | 53/200 | 19.5% | +0.0 | [+0.0, +0.0] | 0 / 0 | 0 |
| sa_paper:1.0:3 | 182/200 | 20.0% | +0.5 | [-2.5, +3.5] | 5 / 4 | 0 |

Δ is paired over the same tasks; the CI is a bootstrap over tasks.

## Run-to-run noise (vLLM, greedy, identical prompts)

| condition | runs | success per run (%) | range |
|---|---|---|---|
| none | 4 | 17.5, 18.5, 18.5, 18.0 | 1.0 pts |
| dg | 3 | 21.0, 22.0, 22.0 | 1.0 pts |

none runs: dg-14b (first), dg-14b-v1-strict-explore (same eval prompts), then repeats. dg runs use the same 133 rules.
