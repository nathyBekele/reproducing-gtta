# Results: qwen2.5-1.5b-bf16 (Qwen/Qwen2.5-1.5B-Instruct)

Examples: 200 (val 40, test 160). Copy-rival α chosen on val: **0.25**.

## Q1: what δ boosts (top-20 tokens of δWᵀ, all examples)

| variant | in prompt | in system | in user | in answer | whitespace+punct | identifiers | mean pairwise Jaccard |
|---|---|---|---|---|---|---|---|
| sa_paper | 73% | 72% | 20% | 12% | 50% | 33% | 0.39 |
| sa_code_mask | 39% | 39% | 4% | 3% | 34% | 45% | 0.11 |
| sa_exact | 72% | 71% | 20% | 11% | 51% | 33% | 0.37 |

Most often boosted (sa_paper): `' in'`×176, `'.'`×173, `' '`×171, `'\n'`×168, `','`×162, `'\n\n'`×162, `'-'`×156, `'1'`×151, `' ('`×146, `'('`×142

Most often boosted (sa_code_mask): `'.'`×73, `'\n\n'`×63, `' '`×62, `','`×60, `'\n'`×58, `' current'`×57, `'current'`×53, `',…\n\n'`×53, `' ('`×52, `'1'`×51

Most often boosted (sa_exact): `'.'`×178, `' in'`×177, `'\n'`×175, `' '`×173, `','`×164, `'\n\n'`×162, `'-'`×158, `'1'`×148, `' ('`×144, `'('`×137

## Q2–Q4: log-prob of the gold answer (test, mean per token; Δ vs. no adaptation)

| condition | mean log-prob | Δ | 95% CI | % examples improved |
|---|---|---|---|---|
| none | -0.8555 | – | – | – |
| sa_paper | -0.8554 | +0.0000 | [-0.0003, +0.0003] | 64% |
| sa_code_mask | -0.8559 | -0.0004 | [-0.0007, -0.0002] | 41% |
| sa_exact | -0.8591 | -0.0037 | [-0.0119, +0.0043] | 56% |
| copy_alpha_0.25 | -0.8547 | +0.0007 | [-0.0009, +0.0023] | 73% |
| random_gaussian_like_sa_paper | -0.8554 | +0.0001 | [+0.0000, +0.0001] | 66% |
| random_sign_like_sa_paper | -0.8554 | +0.0000 | [+0.0000, +0.0001] | 64% |
| random_gaussian_like_sa_code_mask | -0.8554 | +0.0001 | [+0.0000, +0.0001] | 66% |
| random_sign_like_sa_code_mask | -0.8554 | +0.0000 | [+0.0000, +0.0001] | 64% |
| random_gaussian_like_sa_exact | -0.8537 | +0.0017 | [+0.0002, +0.0033] | 65% |
| random_sign_like_sa_exact | -0.8545 | +0.0009 | [-0.0000, +0.0019] | 60% |

## Paired comparisons (test, mean per token)

| A − B | Δ | 95% CI | % A better |
|---|---|---|---|
| sa_exact − copy_alpha_0.25 | -0.0044 | [-0.0120, +0.0029] | 54% |
| sa_exact − random_sign_like_sa_exact | -0.0046 | [-0.0133, +0.0037] | 56% |
| sa_exact − sa_paper | -0.0037 | [-0.0116, +0.0040] | 56% |
| sa_paper − random_sign_like_sa_paper | -0.0000 | [-0.0004, +0.0003] | 64% |

## SA diagnostics (all examples)

| variant | median ‖δ‖ | ‖δ‖ range | median % \|g\|<ε | median Δ prompt loss | Spearman(‖δ‖, prompt length) |
|---|---|---|---|---|---|
| sa_paper | 0.101 | 0.057–0.200 | 100% | -0.00060 | -0.99 |
| sa_code_mask | 0.105 | 0.055–0.204 | 100% | -0.00053 | -0.93 |
| sa_exact | 3.822 | 3.785–3.849 | 1% | -0.00919 | -0.90 |
