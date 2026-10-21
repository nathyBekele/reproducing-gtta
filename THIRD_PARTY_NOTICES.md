# Third-party notices

This repository includes material from two projects. Each keeps its original license.

## GTTA (MIT)
- **What:** the BFCLv3 task files, the 40/160 validation/test split, the exploration goals and the DG prompts in `data/bfcl_v3/`, and the vendored BFCL code in `third_party/bfcl/` as distributed in the GTTA repository.
- **Source:** https://github.com/r2llab/GTTA, commit `24ba3040d3131ca49367ec12898e2150eb31aa5d`
- **License:** MIT. Copyright (c) 2025 Arthur Chen, Zuxin Liu, Jianguo Zhang, Akshara Prabhakar, Zhiwei Liu, Shelby Heinecke, Silvio Savarese, Victor Zhong, Caiming Xiong.

## Berkeley Function-Calling Leaderboard (Apache-2.0)
- **What:** `third_party/bfcl/` (the multi-turn API simulators, `multi_turn_utils`, `multi_turn_checker`, and parsing helpers), which GTTA builds on.
- **Source:** https://github.com/ShishirPatil/gorilla (berkeley-function-call-leaderboard)
- **License:** Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0. The files were copied from GTTA's copy of BFCL (commit above); see `third_party/SOURCE_COMMIT_GTTA`.

## Paper
Chen et al., *Test-Time Adaptation for LLM Agents via Environment Interaction*, ICLR 2026, https://arxiv.org/abs/2511.04847. This repository is an independent reproduction and is not affiliated with the authors.
