### Table 1: Retrieval Performance on FinQA Test Split (n = 1,147)

| Method | R@1 (%) | R@3 (%) | R@5 (%) | Full@5 (%) | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| BM25 ($k_1=1.5, b=0.75$) | 52.92 | 77.24 | 87.01 | 62.51 | 0.6726 |
| TF-IDF Cosine | 48.30 | 73.50 | 83.44 | 59.72 | 0.6317 |
| Hybrid Fusion ($\alpha=0.5$) | 51.61 | 77.16 | 86.40 | 62.86 | 0.6631 |
| **Hybrid + GBT Reranker** | 79.95 | 93.81 | 97.47 | 80.56 | 0.8752 |

### Table 2: End-to-End Accuracy, Ablations & Compliance Gate (FinQA Test)

_Compliance threshold tuned on FinQA dev: $\tau = 0.1300$_

| System Variant | Overall Acc (%) | Selective Acc (%) | 95% Bootstrap CI | McNemar $p$ vs. Baseline | Coverage (%) | CPU ms/query |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Single-agent baseline | 4.97 | 5.07 | [3.84, 6.28] | — | 98.08 | 22.83 |
| AUGUS w/o reranker | 19.70 | 28.61 | [17.52, 22.06] | 1.25e-37 | 68.88 | 19.95 |
| AUGUS BM25-only | 25.11 | 35.82 | [22.58, 27.55] | 1.84e-51 | 70.10 | 33.77 |
| AUGUS w/o planner | 7.85 | 7.85 | [6.45, 9.33] | 3.76e-05 | 100.00 | 41.25 |
| AUGUS full | 25.11 | 35.82 | [22.58, 27.55] | 1.84e-51 | 70.10 | 36.38 |
| Gold-evidence oracle | 32.26 | 47.01 | [29.47, 34.87] | 3.97e-77 | 68.61 | 2249.08 |
| AUGUS + Compliance gate | 14.56 | 60.51 | [12.64, 16.56] | 1.60e-17 | 24.06 | 3233.84 |

### Table 3: Falsified-Operand Grounding Stress Test

| Metric | Value |
| :--- | :---: |
| Executable Test Programs Evaluated | 821 |
| Perturbed Operands Checked ($\times \{0.9, 0.95, 0.99, 1.01, 1.05, 1.1\}$) | 819 |
| Perturbation Detection Rate (Top-5 Evidence) | 100.00% |
| Perturbation Detection Rate (Full Filing) | 100.00% |
| False-Alarm Rate on Clean Gold Operands (Top-5) | 16.08% |
| False-Alarm Rate on Clean Gold Operands (Full Filing) | 8.77% |
