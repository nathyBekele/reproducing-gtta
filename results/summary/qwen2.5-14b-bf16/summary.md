# Results: qwen2.5-14b-bf16 (Qwen/Qwen2.5-14B-Instruct)

Examples: 200 (val 40, test 160). Copy-rival α chosen on val: **0.25**.

## Q1: what δ boosts (top-20 tokens of δWᵀ, all examples)

| variant | in prompt | in system | in user | in answer | whitespace+punct | identifiers | mean pairwise Jaccard |
|---|---|---|---|---|---|---|---|
| sa_paper | 75% | 73% | 17% | 8% | 45% | 45% | 0.29 |
| sa_code_mask | 41% | 41% | 3% | 3% | 11% | 58% | 0.14 |
| sa_exact | 70% | 70% | 15% | 7% | 37% | 52% | 0.24 |

Most often boosted (sa_paper): `'.'`×190, `'\n'`×185, `' '`×155, `','`×155, `'.\n'`×151, `'('`×117, `'-'`×114, `'1'`×111, `'['`×110, `'?'`×106

Most often boosted (sa_code_mask): `'.'`×114, `'float'`×98, `'dict'`×92, `'current'`×91, `'Current'`×81, `'_current'`×74, `' Current'`×70, `' current'`×64, `' Note'`×56, `'ᐈ'`×55

Most often boosted (sa_exact): `'.'`×182, `'\n'`×178, `' '`×147, `','`×139, `'.\n'`×108, `'('`×106, `'1'`×101, `' Note'`×100, `'Note'`×100, `' note'`×100

## Q2–Q4: log-prob of the gold answer (test, mean per token; Δ vs. no adaptation)

| condition | mean log-prob | Δ | 95% CI | % examples improved |
|---|---|---|---|---|
| none | -1.2628 | – | – | – |
| sa_paper | -1.2630 | -0.0002 | [-0.0004, +0.0000] | 59% |
| sa_code_mask | -1.2631 | -0.0003 | [-0.0005, -0.0001] | 45% |
| sa_exact | -1.2736 | -0.0108 | [-0.0212, -0.0011] | 52% |
| copy_alpha_0.25 | -1.2635 | -0.0007 | [-0.0022, +0.0007] | 59% |
| random_gaussian_like_sa_paper | -1.2628 | +0.0000 | [-0.0000, +0.0000] | 41% |
| random_sign_like_sa_paper | -1.2628 | -0.0000 | [-0.0000, +0.0000] | 32% |
| random_gaussian_like_sa_code_mask | -1.2628 | +0.0000 | [-0.0000, +0.0000] | 41% |
| random_sign_like_sa_code_mask | -1.2628 | -0.0000 | [-0.0000, +0.0000] | 32% |
| random_gaussian_like_sa_exact | -1.2626 | +0.0003 | [-0.0010, +0.0017] | 39% |
| random_sign_like_sa_exact | -1.2636 | -0.0007 | [-0.0018, +0.0005] | 32% |

## Paired comparisons (test, mean per token)

| A − B | Δ | 95% CI | % A better |
|---|---|---|---|
| sa_exact − copy_alpha_0.25 | -0.0101 | [-0.0200, -0.0010] | 52% |
| sa_exact − random_sign_like_sa_exact | -0.0101 | [-0.0209, -0.0001] | 52% |
| sa_exact − sa_paper | -0.0106 | [-0.0208, -0.0012] | 52% |
| sa_paper − random_sign_like_sa_paper | -0.0002 | [-0.0004, +0.0001] | 61% |

## SA diagnostics (all examples)

| variant | median ‖δ‖ | ‖δ‖ range | median % \|g\|<ε | median Δ prompt loss | Spearman(‖δ‖, prompt length) |
|---|---|---|---|---|---|
| sa_paper | 0.108 | 0.065–0.238 | 100% | -0.00068 | -0.99 |
| sa_code_mask | 0.115 | 0.062–0.234 | 100% | -0.00064 | -0.96 |
| sa_exact | 6.888 | 6.830–6.959 | 1% | -0.02115 | -0.94 |
