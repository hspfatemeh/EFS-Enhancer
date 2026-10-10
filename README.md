## Reproducing the results

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hspfatemeh/EFS-Enhancer/blob/main/EFS_Enhancer_Colab.ipynb)

**Google Colab:** open the notebook above and select *Runtime → Run all*. No runtime restart is needed.

**Local:**
```bash
git clone https://github.com/hspfatemeh/EFS-Enhancer.git
cd EFS-Enhancer
pip install uv
uv venv -p 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python run_all.py
```
(The pinned versions, e.g. numpy 1.26.4, require Python 3.9–3.12; with Python 3.11 already installed, `pip install -r requirements.txt && python run_all.py` also works.)

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
