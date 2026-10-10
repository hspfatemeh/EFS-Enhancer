# EFS-Enhancer

**EFS-Enhancer: Identifying Enhancer Sequences Based on Deep Learning with Emphasis on Effective Feature Selection**

EFS-Enhancer is a deep learning framework for enhancer sequence identification that combines biologically meaningful feature extraction. The framework is designed to improve prediction performance while reducing feature redundancy and enhancing model interpretability.

---

## Overview

Enhancers are short non-coding DNA sequences that regulate gene expression. Accurate identification of enhancer sequences is essential for understanding gene regulation, disease mechanisms, and functional genomics.

EFS-Enhancer introduces a hybrid pipeline that integrates feature engineering, statistical filtering, feature selection, and machine learning to distinguish enhancer sequences from non-enhancer sequences.

---

## Reproducing the results

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hspfatemeh/EFS-Enhancer/blob/main/EFS_Enhancer_Colab.ipynb)

**Google Colab:** open the notebook above and select *Runtime → Run all*. No runtime restart is needed.

**Local** (Python 3.10–3.12):
```bash
git clone https://github.com/hspfatemeh/EFS-Enhancer.git
cd EFS-Enhancer
pip install -r requirements.txt
python run_all.py
```

`run_all.py` does the following:
- downloads the benchmark (training) and independent (test) datasets from iEnhancer-RD (pinned commit `809aecc`);
- trains both XGBoost layers with the final hyperparameters, which were selected by 5-fold `GridSearchCV` on the benchmark set only;
- evaluates both layers on the independent set, both as given and after removing the independent sequences that are identical to benchmark sequences;
- computes the motif statistics (Fisher's exact test with Benjamini–Hochberg FDR correction) and reproduces all figures.

| Setting | Value |
|---|---|
| Random seeds | 42 |
| Layer 1 | `n_estimators=150, learning_rate=0.2, max_depth=20, min_child_weight=4`, DNA2Vec 4-mers |
| Layer 2 | `n_estimators=40, learning_rate=0.05, max_depth=10, min_child_weight=8`, DNA2Vec 3-mers, feature-importance threshold 0.002 |

Everything is written to `results/`:
- figures, as PDF and PNG
- motif tables, as CSV
- `summary_metrics.csv`: all performance metrics
- `environment.txt`: package versions and hardware

GitHub: https://github.com/hspfatemeh
