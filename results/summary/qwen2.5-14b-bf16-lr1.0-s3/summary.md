# Results: qwen2.5-14b-bf16-lr1.0-s3 (Qwen/Qwen2.5-14B-Instruct)

Examples: 200 (val 40, test 160). Copy-rival α chosen on val: **0.25**.

## Q1: what δ boosts (top-20 tokens of δWᵀ, all examples)

| variant | in prompt | in system | in user | in answer | whitespace+punct | identifiers | mean pairwise Jaccard |
|---|---|---|---|---|---|---|---|
| sa_paper | 67% | 66% | 12% | 6% | 23% | 65% | 0.19 |
| sa_code_mask | 38% | 38% | 3% | 2% | 7% | 63% | 0.13 |
| sa_exact | 16% | 16% | 2% | 1% | 34% | 38% | 0.11 |

Most often boosted (sa_paper): `'.'`×163, `'\n'`×145, `'dict'`×108, `' Note'`×100, `' note'`×100, `'Note'`×100, `' current'`×95, `'current'`×92, `'.\n'`×89, `'float'`×81

Most often boosted (sa_code_mask): `'dict'`×100, `'float'`×98, `'.'`×94, `'current'`×91, `'Current'`×83, `'_current'`×73, `' Current'`×69, `' current'`×65, `'-current'`×58, `' Note'`×56

Most often boosted (sa_exact): `'监听页面'`×93, `'Intialized'`×90, `'صند'`×89, `"'):"`×87, `' belongs'`×81, `"':"`×81, `"']:"`×79, `'ᐈ'`×74, `" ':"`×70, `'GuidId'`×67

## Q2–Q4: log-prob of the gold answer (test, mean per token; Δ vs. no adaptation)

| condition | mean log-prob | Δ | 95% CI | % examples improved |
|---|---|---|---|---|
| none | -1.2650 | – | – | – |
| sa_paper | -1.2677 | -0.0027 | [-0.0077, +0.0019] | 57% |
| sa_code_mask | -1.2710 | -0.0060 | [-0.0110, -0.0016] | 47% |
| sa_exact | -1.7152 | -0.4502 | [-0.5734, -0.3310] | 36% |
| copy_alpha_0.25 | -1.2657 | -0.0006 | [-0.0022, +0.0008] | 59% |
| random_gaussian_like_sa_paper | -1.2649 | +0.0001 | [-0.0006, +0.0008] | 41% |
| random_sign_like_sa_paper | -1.2654 | -0.0004 | [-0.0010, +0.0002] | 33% |
| random_gaussian_like_sa_code_mask | -1.2649 | +0.0001 | [-0.0006, +0.0008] | 41% |
| random_sign_like_sa_code_mask | -1.2655 | -0.0004 | [-0.0010, +0.0002] | 33% |
| random_gaussian_like_sa_exact | -1.2883 | -0.0232 | [-0.0387, -0.0071] | 24% |
| random_sign_like_sa_exact | -1.2955 | -0.0304 | [-0.0446, -0.0161] | 19% |

## Paired comparisons (test, mean per token)

| A − B | Δ | 95% CI | % A better |
|---|---|---|---|
| sa_exact − copy_alpha_0.25 | -0.4496 | [-0.5729, -0.3298] | 36% |
| sa_exact − random_sign_like_sa_exact | -0.4198 | [-0.5454, -0.2992] | 36% |
| sa_exact − sa_paper | -0.4475 | [-0.5700, -0.3287] | 36% |
| sa_paper − random_sign_like_sa_paper | -0.0023 | [-0.0073, +0.0024] | 58% |

## SA diagnostics (all examples)

| variant | median ‖δ‖ | ‖δ‖ range | median % \|g\|<ε | median Δ prompt loss | Spearman(‖δ‖, prompt length) |
|---|---|---|---|---|---|
| sa_paper | 2.978 | 1.858–5.779 | 100% | -0.01558 | -0.99 |
| sa_code_mask | 3.191 | 1.804–5.968 | 100% | -0.01433 | -0.96 |
| sa_exact | 83.374 | 78.645–93.268 | 1% | +0.78823 | -0.15 |
