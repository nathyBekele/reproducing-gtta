# Results: qwen2.5-14b-bf16-lr1.0-s1 (Qwen/Qwen2.5-14B-Instruct)

Examples: 200 (val 40, test 160). Copy-rival α chosen on val: **0.25**.

## Q1: what δ boosts (top-20 tokens of δWᵀ, all examples)

| variant | in prompt | in system | in user | in answer | whitespace+punct | identifiers | mean pairwise Jaccard |
|---|---|---|---|---|---|---|---|
| sa_paper | 74% | 73% | 17% | 8% | 45% | 46% | 0.28 |
| sa_code_mask | 41% | 41% | 3% | 3% | 11% | 58% | 0.14 |
| sa_exact | 70% | 70% | 15% | 7% | 37% | 52% | 0.24 |

Most often boosted (sa_paper): `'.'`×190, `'\n'`×185, `' '`×154, `','`×153, `'.\n'`×150, `'('`×118, `'-'`×111, `'1'`×110, `'['`×110, `'?'`×104

Most often boosted (sa_code_mask): `'.'`×114, `'float'`×98, `'dict'`×92, `'current'`×90, `'Current'`×82, `'_current'`×75, `' Current'`×71, `' current'`×63, `' Note'`×56, `'ᐈ'`×55

Most often boosted (sa_exact): `'.'`×181, `'\n'`×177, `' '`×146, `','`×141, `'.\n'`×108, `'('`×106, `' Note'`×100, `'Note'`×100, `' note'`×100, `'1'`×99

## Q2–Q4: log-prob of the gold answer (test, mean per token; Δ vs. no adaptation)

| condition | mean log-prob | Δ | 95% CI | % examples improved |
|---|---|---|---|---|
| none | -1.2650 | – | – | – |
| sa_paper | -1.2670 | -0.0019 | [-0.0046, +0.0004] | 55% |
| sa_code_mask | -1.2682 | -0.0032 | [-0.0057, -0.0010] | 43% |
| sa_exact | -1.7026 | -0.4376 | [-0.5664, -0.3186] | 21% |
| copy_alpha_0.25 | -1.2657 | -0.0006 | [-0.0022, +0.0008] | 59% |
| random_gaussian_like_sa_paper | -1.2650 | +0.0000 | [-0.0002, +0.0003] | 41% |
| random_sign_like_sa_paper | -1.2652 | -0.0001 | [-0.0004, +0.0001] | 32% |
| random_gaussian_like_sa_code_mask | -1.2650 | +0.0000 | [-0.0002, +0.0003] | 41% |
| random_sign_like_sa_code_mask | -1.2652 | -0.0001 | [-0.0004, +0.0001] | 32% |
| random_gaussian_like_sa_exact | -1.2797 | -0.0146 | [-0.0272, -0.0015] | 26% |
| random_sign_like_sa_exact | -1.2866 | -0.0215 | [-0.0331, -0.0097] | 19% |

## Paired comparisons (test, mean per token)

| A − B | Δ | 95% CI | % A better |
|---|---|---|---|
| sa_exact − copy_alpha_0.25 | -0.4370 | [-0.5651, -0.3184] | 21% |
| sa_exact − random_sign_like_sa_exact | -0.4161 | [-0.5466, -0.2972] | 22% |
| sa_exact − sa_paper | -0.4356 | [-0.5626, -0.3186] | 21% |
| sa_paper − random_sign_like_sa_paper | -0.0018 | [-0.0044, +0.0005] | 60% |

## SA diagnostics (all examples)

| variant | median ‖δ‖ | ‖δ‖ range | median % \|g\|<ε | median Δ prompt loss | Spearman(‖δ‖, prompt length) |
|---|---|---|---|---|---|
| sa_paper | 1.076 | 0.653–2.384 | 100% | -0.00632 | -0.99 |
| sa_code_mask | 1.153 | 0.624–2.344 | 100% | -0.00559 | -0.96 |
| sa_exact | 68.878 | 68.298–69.620 | 1% | +1.40131 | -0.94 |
