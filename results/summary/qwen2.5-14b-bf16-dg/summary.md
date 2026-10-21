# Results: qwen2.5-14b-bf16-dg (Qwen/Qwen2.5-14B-Instruct)

Examples: 200 (val 40, test 160). Copy-rival α chosen on val: **0.25**.

## Q1: what δ boosts (top-20 tokens of δWᵀ, all examples)

| variant | in prompt | in system | in user | in answer | whitespace+punct | identifiers | mean pairwise Jaccard |
|---|---|---|---|---|---|---|---|
| sa_paper | 73% | 72% | 17% | 10% | 38% | 45% | 0.27 |
| sa_code_mask | 43% | 42% | 3% | 3% | 11% | 56% | 0.16 |
| sa_exact | 70% | 69% | 15% | 8% | 30% | 54% | 0.23 |

Most often boosted (sa_paper): `'.'`×193, `'\n'`×183, `' environment'`×155, `'环境'`×138, `'1'`×130, `'.\n'`×124, `','`×113, `'2'`×110, `'-'`×102, `'?'`×93

Most often boosted (sa_code_mask): `'环境'`×172, `' environment'`×131, `'.'`×109, `'環境'`×108, `'dict'`×107, `'environment'`×105, `'环境中'`×97, `'float'`×84, `' Environment'`×80, `'current'`×73

Most often boosted (sa_exact): `'.'`×192, `'\n'`×169, `' environment'`×159, `'环境'`×153, `'1'`×116, `' Note'`×96, `'Note'`×93, `','`×92, `' Environment'`×85, `' '`×83

## Q2–Q4: log-prob of the gold answer (test, mean per token; Δ vs. no adaptation)

| condition | mean log-prob | Δ | 95% CI | % examples improved |
|---|---|---|---|---|
| none | -1.3865 | – | – | – |
| sa_paper | -1.3867 | -0.0002 | [-0.0003, -0.0001] | 40% |
| sa_code_mask | -1.3868 | -0.0003 | [-0.0004, -0.0002] | 36% |
| sa_exact | -1.4041 | -0.0176 | [-0.0266, -0.0088] | 34% |
| copy_alpha_0.25 | -1.3836 | +0.0029 | [+0.0007, +0.0050] | 60% |
| random_gaussian_like_sa_paper | -1.3865 | -0.0000 | [-0.0000, +0.0000] | 36% |
| random_sign_like_sa_paper | -1.3865 | +0.0000 | [-0.0000, +0.0000] | 38% |
| random_gaussian_like_sa_code_mask | -1.3865 | -0.0000 | [-0.0000, +0.0000] | 36% |
| random_sign_like_sa_code_mask | -1.3865 | +0.0000 | [-0.0000, +0.0000] | 38% |
| random_gaussian_like_sa_exact | -1.3867 | -0.0002 | [-0.0015, +0.0014] | 34% |
| random_sign_like_sa_exact | -1.3868 | -0.0002 | [-0.0015, +0.0013] | 36% |

## Paired comparisons (test, mean per token)

| A − B | Δ | 95% CI | % A better |
|---|---|---|---|
| sa_exact − copy_alpha_0.25 | -0.0205 | [-0.0302, -0.0112] | 35% |
| sa_exact − random_sign_like_sa_exact | -0.0174 | [-0.0262, -0.0088] | 32% |
| sa_exact − sa_paper | -0.0174 | [-0.0263, -0.0087] | 34% |
| sa_paper − random_sign_like_sa_paper | -0.0002 | [-0.0003, -0.0001] | 41% |

## SA diagnostics (all examples)

| variant | median ‖δ‖ | ‖δ‖ range | median % \|g\|<ε | median Δ prompt loss | Spearman(‖δ‖, prompt length) |
|---|---|---|---|---|---|
| sa_paper | 0.070 | 0.047–0.171 | 100% | -0.00037 | -0.98 |
| sa_code_mask | 0.070 | 0.046–0.175 | 100% | -0.00036 | -0.95 |
| sa_exact | 6.858 | 6.797–6.938 | 1% | -0.01851 | -0.91 |
