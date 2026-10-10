"""EFS-Enhancer: full reproduction of the paper results.

Usage (Colab or local):
    pip install -r requirements.txt
    python run_all.py
All figures (PDF + PNG), tables (CSV), summary_metrics.csv and environment.txt are written to results/.
"""
import os, sys, subprocess, urllib.request, importlib.metadata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

PINNED = {"numpy": "1.26.4", "xgboost": "2.1.4", "shap": "0.45.1", "scikit-learn": "1.6.1", "pandas": "2.2.2",
          "scipy": "1.15.3", "statsmodels": "0.14.5", "biopython": "1.88", "matplotlib": "3.10.0",
          "seaborn": "0.13.2", "logomaker": "0.8.7"}
def pkg_version(p):
    try:
        return importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
for p, v in PINNED.items():
    got = pkg_version(p)
    if got != v:
        print(f"WARNING: {p}=={got} (paper used {v}); results may differ slightly")

# ---- data: benchmark (training) and independent (test) sets from iEnhancer-RD, pinned commit ----
RD_COMMIT = "809aeccdaeb05c7b69e0938d49cceec2ce4a97f0"
if not os.path.isdir("iEnhancer-RD"):
    subprocess.run(["git", "clone", "-q", "https://github.com/YangHuan639/iEnhancer-RD.git"], check=True)
    subprocess.run(["git", "-C", "iEnhancer-RD", "checkout", "-q", RD_COMMIT], check=True)
DATA = "iEnhancer-RD/dataset/"

# ---- DNA2Vec embeddings (3-5-mers, 100-d) shipped with this repository ----
W2V = "dna2vec-20251027-1738-k3to5-100d-2c-0Mbp-sliding-338.w2v"
if not os.path.exists(W2V):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/hspfatemeh/EFS-Enhancer/main/" + W2V, W2V)

OUT = "results"
os.makedirs(OUT, exist_ok=True)

# every figure is saved as PDF (paper) and PNG (quick viewing); nothing is shown interactively
_savefig = plt.savefig
def _save_both(path, *a, **k):
    _savefig(path, *a, **k)
    if str(path).endswith(".pdf"):
        k = {kk: vv for kk, vv in k.items() if kk not in ("format", "dpi")}
        _savefig(str(path)[:-4] + ".png", dpi=150, **k)
plt.savefig = _save_both
plt.show = lambda *a, **k: plt.close("all")

def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x)

def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78, flush=True)


# ---- 3. Imports, seeds, shared functions ----
import random, platform, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, logomaker, shap
from collections import Counter
from itertools import product
from scipy.stats import entropy, fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import Table2x2
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from Bio.Seq import Seq
from Bio import motifs as bio_motifs
warnings.filterwarnings("ignore")

random.seed(42); np.random.seed(42); os.environ["PYTHONHASHSEED"] = "42"

MOTIFS = ['TATA', 'CCCTG', 'CCTGG', 'GCCTG', 'CCCAGG', 'CCCAGCC', 'CTCCTCC']
SUMMARY = []

def read_fasta(path):
    seqs, seq = [], ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq: seqs.append(seq); seq = ""
            else:
                seq += line
    if seq: seqs.append(seq)
    return seqs

def get_all_kmers(k): return [''.join(p) for p in product('ACGT', repeat=k)]

# DNA2Vec sequence embedding (mean of k-mer vectors)
def load_dna2vec(path, k):
    d = {}
    with open(path) as f:
        for line in f:
            v = line.strip().split()
            if len(v[0]) == k: d[v[0]] = np.array(v[1:], dtype=np.float32)
    return d

def kmer_tokenize_str(seqs, k): return [[s.upper()[i:i+k] for i in range(len(s)-k+1)] for s in seqs]

def build_kmer_index(kmer_seqs):
    return {km: i+1 for i, km in enumerate(sorted(set(km for s in kmer_seqs for km in s)))}

def encode_kmers(kmer_seqs, idx):
    enc = np.zeros((len(kmer_seqs), len(kmer_seqs[0])), dtype=np.int32)
    for i, s in enumerate(kmer_seqs):
        for j, km in enumerate(s): enc[i, j] = idx.get(km, 0)
    return enc

def build_embedding_matrix(idx, emb, dim):
    M = np.zeros((len(idx) + 1, dim))
    for km, i in idx.items():
        M[i] = emb[km] if km in emb else np.random.normal(scale=0.1, size=(dim,))
    return M

def dna2vec_features(train, test, k, dim=100):
    ks = kmer_tokenize_str(train, k); idx = build_kmer_index(ks)
    M = build_embedding_matrix(idx, load_dna2vec(W2V, k), dim)
    avg = lambda enc: np.array([M[s].mean(axis=0) for s in enc])
    return avg(encode_kmers(ks, idx)), avg(encode_kmers(kmer_tokenize_str(test, k), idx))

# handcrafted features
def kmer_frequency_features(seqs, k=3):
    out = []
    for s in seqs:
        c = Counter(s.upper()[i:i+k] for i in range(len(s)-k+1)); t = sum(c.values())
        out.append([c[km]/t if t else 0.0 for km in get_all_kmers(k)])
    return np.array(out)

def extract_stat_features(seqs, k=3):   # GC content, Shannon entropy (natural log) of 3-mers
    out = []
    for s in seqs:
        s = s.upper(); km = [s[i:i+k] for i in range(len(s)-k+1)]
        out.append([(s.count('G') + s.count('C'))/len(s), entropy([v/len(km) for v in Counter(km).values()])])
    return np.array(out)

def pwm_score_features(seqs, motif_list):
    lo = [bio_motifs.create([Seq(m)]).counts.normalize(pseudocounts=0.1).log_odds() for m in motif_list]
    out = []
    for s in seqs:
        s = Seq(s.upper())
        out.append([max([sc for _, sc in l.search(s)], default=0.0) for l in lo])
    return np.array(out)

def dinuc_freq(seqs):
    d = [a+b for a in 'ACGT' for b in 'ACGT']; out = []
    for s in seqs:
        s = s.upper(); t = len(s) - 1; c = Counter(s[i:i+2] for i in range(t))
        out.append([c[x]/t if t else 0 for x in d])
    return np.array(out)

def build_features(train, test, k_dna2vec):
    Xtr, Xte = dna2vec_features(train, test, k_dna2vec)
    for f in (lambda s: kmer_frequency_features(s, 3), lambda s: extract_stat_features(s, 3),
              lambda s: pwm_score_features(s, MOTIFS), dinuc_freq):
        Xtr = np.concatenate([Xtr, f(train)], axis=1); Xte = np.concatenate([Xte, f(test)], axis=1)
    names = ([f"dna2vec_dim{i}" for i in range(100)] + get_all_kmers(3) + ["GC_content", "Entropy"]
             + MOTIFS + [a+b for a in 'ACGT' for b in 'ACGT'])
    return Xtr, Xte, names

# model and evaluation
def performance(y, p):
    TN, FP, FN, TP = metrics.confusion_matrix(y, p, labels=[0, 1]).ravel()
    return (metrics.accuracy_score(y, p), TP/(TP+FN) if TP+FN else np.nan,
            TN/(TN+FP) if TN+FP else np.nan, matthews_corrcoef(y, p))

def benchmark_train(X, y, param_grid):
    gs = GridSearchCV(XGBClassifier(eval_metric='logloss'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X, y)
    m = gs.best_estimator_
    return m, m.feature_importances_, shap.TreeExplainer(m), gs.best_score_

def report(layer, setting, y, p, cv=np.nan):
    acc, sn, sp, mcc = performance(y, p)
    SUMMARY.append(dict(Layer=layer, Setting=setting, n=len(y), CV_ACC=cv, ACC=acc, SN=sn, SP=sp, MCC=mcc))
    print(f"{layer} | {setting:38s} n={len(y):3d}  ACC={acc:.4f} SN={sn:.4f} SP={sp:.4f} MCC={mcc:.4f}")

def dedup_mask(train, test):
    tr = set(s.upper() for s in train)
    return np.array([s.upper() not in tr for s in test])

# figures
def motif_enrichment(pos, neg, layer, tag):
    print("Fisher's exact test on all 5-7-mers + TATA ...")
    rows = []
    for m in [mm for k in (5, 6, 7) for mm in get_all_kmers(k)] + ["TATA"]:
        a = sum(m in s for s in pos); c = sum(m in s for s in neg)
        t = np.array([[a, len(pos)-a], [c, len(neg)-c]])
        orr, p = fisher_exact(t)
        lo, hi = Table2x2(t).oddsratio_confint()
        rows.append(dict(Motif=m, Length=len(m), Pos_with=a, Neg_with=c, OddsRatio=orr, OR_CI_low=lo, OR_CI_high=hi, Pvalue=p))
    df = pd.DataFrame(rows)
    kmer = df.Motif != "TATA"   # BH over the 21,504 5-7-mers; TATA's q from the same family + TATA
    df.loc[kmer, "FDR_Pvalue"] = multipletests(df.Pvalue[kmer], method="fdr_bh")[1]
    df.loc[~kmer, "FDR_Pvalue"] = multipletests(df.Pvalue, method="fdr_bh")[1][~kmer.values]
    df.sort_values("FDR_Pvalue").to_csv(f"{OUT}/Motif_Statistics_All_{tag}.csv", index=False)
    sel = df.set_index("Motif").loc[MOTIFS].reset_index()
    sel.to_csv(f"{OUT}/Motif_Statistics_ModelMotifs_{tag}.csv", index=False)
    display(sel)
    x, y = np.log2(sel.OddsRatio), -np.log10(sel.FDR_Pvalue)
    plt.figure(figsize=(8, 6)); plt.scatter(x, y, s=90, c=y, cmap="plasma", edgecolor="k")
    plt.margins(x=0.12); plt.axvline(0, color="red", ls="--", alpha=.7)
    off = {"CCCAGCC": (0.05, -3.0), "CTCCTCC": (0.05, 1.0)} if tag == "1" else {}   # avoid overlapping labels
    for xi, yi, m in zip(x, y, MOTIFS):
        dx, dy = off.get(m, (0.05, 0)); plt.text(xi + dx, yi + dy, m, fontsize=9)
    plt.xlabel(r"$\log_2(\mathrm{Odds\ Ratio})$"); plt.ylabel(r"$-\log_{10}(\mathrm{FDR-adjusted\ p-value})$")
    plt.title(f"Motif Enrichment - {layer}"); plt.colorbar(label=r"$-\log_{10}(\mathrm{FDR-adjusted\ p-value})$")
    plt.tight_layout(); plt.savefig(f"{OUT}/MotifLayer{tag}_FDR.pdf", bbox_inches="tight"); plt.show()

def gc_plot(seqs, labels, names, layer, fname):
    st = extract_stat_features(seqs)
    df = pd.DataFrame(dict(GC_content=st[:, 0], Entropy=st[:, 1], Label=[names[l] for l in labels]))
    plt.figure(figsize=(8, 6)); sns.scatterplot(data=df, x='GC_content', y='Entropy', hue='Label', alpha=.6)
    plt.title(f'GC Content vs Sequence Entropy - {layer}'); plt.xlabel('GC Content'); plt.ylabel('Entropy')
    plt.savefig(f"{OUT}/{fname}", bbox_inches="tight"); plt.show()
    for l, n in names.items():
        print(f"  Pearson r(GC, entropy) {n}: {np.corrcoef(st[np.array(labels)==l].T)[0,1]:.3f}")

def importance_plots(imp, shap_values, names, layer, suffix, top_n=33):
    pats = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    for vals, ylab, kind in ((imp, "Feature Importance", "XGBoost"),
                             (np.abs(shap_values).mean(axis=0), "Mean |SHAP value|", "SHAP")):
        idx = np.argsort(vals)[::-1][:top_n]
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(top_n), vals[idx], color="white", edgecolor="black", width=.6)
        for i, b in enumerate(bars): b.set_hatch(pats[i % len(pats)])
        plt.xticks(range(top_n), np.array(names)[idx], rotation=90); plt.ylabel(ylab, fontsize=12)
        plt.title(f"Top {top_n} Feature Importances ({kind}) - {layer}", fontsize=14)
        plt.grid(axis='y', ls='--', alpha=.5); plt.tight_layout()
        plt.savefig(f"{OUT}/Top_Feature_Importances_{kind}_Patterned_Vertical{suffix}.pdf", bbox_inches="tight"); plt.show()

def sequence_logo(seqs, fname, min_count=368):
    L = len(seqs[0]); assert all(len(s) == L for s in seqs)
    df = pd.DataFrame({b: [sum(s[i] == b for s in seqs) for i in range(L)] for b in "ACGT"})
    df[df < min_count] = 0
    ppm = df.div(df.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(14, 1.2)); logomaker.Logo(ppm, vpad=.09, width=.9, ax=ax)
    ax.set_xlabel("Sequence Position"); ax.set_ylabel("Frequency"); plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=300, bbox_inches="tight"); plt.show()

section("LAYER 1: enhancer vs non-enhancer")
pos, neg = read_fasta(DATA + "B_Enhancer.txt"), read_fasta(DATA + "B_NonEnhancer.txt")
tpos, tneg = read_fasta(DATA + "I_Enhancer.txt"), read_fasta(DATA + "I_NonEnhancer.txt")
train1, y1 = [s.upper() for s in pos + neg], np.array([1]*len(pos) + [0]*len(neg))
test1, yt1 = [s.upper() for s in tpos + tneg], np.array([1]*len(tpos) + [0]*len(tneg))
print(f"Benchmark (training): {len(pos)} enhancers / {len(neg)} non-enhancers | Independent (test): {len(tpos)} / {len(tneg)}")

sequence_logo([s.upper() for s in pos], "B_Enhancer.pdf"); sequence_logo([s.upper() for s in neg], "B_NonEnhancer.pdf")
motif_enrichment(pos, neg, "Layer 1", "1")
gc_plot(train1, y1, {1: "Enhancer", 0: "Non-Enhancer"}, "layer 1", "GXContentLayer1.pdf")

PARAMS_L1 = {'n_estimators': [150], 'learning_rate': [0.2], 'max_depth': [20], 'min_child_weight': [4]}
X1, Xt1, names1 = build_features(train1, test1, k_dna2vec=4)
print("Feature matrix:", X1.shape, Xt1.shape)
model1, imp1, expl1, cv1 = benchmark_train(X1, y1, PARAMS_L1)
p1 = model1.predict(Xt1)
report("Layer 1", "Independent test", yt1, p1, cv1)
keep1 = dedup_mask(train1, test1)
print(f"Independent sequences identical to benchmark sequences: {(~keep1).sum()}")
report("Layer 1", "Independent test, overlap removed", yt1[keep1], p1[keep1])
importance_plots(imp1, expl1.shap_values(Xt1), names1, "Layer 1", "")

section("LAYER 2: strong vs weak enhancer")
pos, neg = read_fasta(DATA + "B_StrongEnhancer.txt"), read_fasta(DATA + "B_WeakEnhancer.txt")
tpos, tneg = read_fasta(DATA + "I_StrongEnhancer.txt"), read_fasta(DATA + "I_WeakEnhancer.txt")
train2, y2 = [s.upper() for s in pos + neg], np.array([1]*len(pos) + [0]*len(neg))
test2, yt2 = [s.upper() for s in tpos + tneg], np.array([1]*len(tpos) + [0]*len(tneg))
print(f"Benchmark (training): {len(pos)} strong / {len(neg)} weak | Independent (test): {len(tpos)} / {len(tneg)}")

motif_enrichment(pos, neg, "Layer 2", "2")
gc_plot(train2, y2, {1: "Strong Enhancer", 0: "Weak Enhancer"}, "layer 2", "GXContentLayer2.pdf")

PARAMS_L2 = {'n_estimators': [40], 'learning_rate': [0.05], 'max_depth': [10], 'min_child_weight': [8]}
X2, Xt2, names2 = build_features(train2, test2, k_dna2vec=3)
model2, imp2, expl2, cv2 = benchmark_train(X2, y2, PARAMS_L2)
report("Layer 2", "Independent test, all features", yt2, model2.predict(Xt2), cv2)
importance_plots(imp2, expl2.shap_values(Xt2), names2, "Layer 2", "2")

# feature selection on training-set importance (threshold 0.002), then retrain
sel = imp2 >= 0.002
print(f"Feature selection: {X2.shape[1]} -> {sel.sum()} features")
model2s, _, _, cv2s = benchmark_train(X2[:, sel], y2, PARAMS_L2)
p2 = model2s.predict(Xt2[:, sel])
report("Layer 2", "Independent test, after feature selection", yt2, p2, cv2s)

keep2 = dedup_mask(train2, test2)
print(f"Independent sequences identical to benchmark sequences: {(~keep2).sum()} "
      f"(strong: {(~keep2 & (yt2==1)).sum()}, weak: {(~keep2 & (yt2==0)).sum()})")
report("Layer 2", "Overlap removed (weak only -> SP only)", yt2[keep2], p2[keep2])

section("SUMMARY")
summary = pd.DataFrame(SUMMARY)
summary.to_csv(f"{OUT}/summary_metrics.csv", index=False)
display(summary.round(4))

import psutil
pkgs = {p: pkg_version(p) for p in PINNED}
env = [f"Python {sys.version.split()[0]} | {platform.platform()}",
       f"CPU: {platform.processor() or platform.machine()} | logical cores: {os.cpu_count()} | "
       f"RAM: {psutil.virtual_memory().total/2**30:.1f} GB"] + [f"{k}=={v}" for k, v in pkgs.items()]
open(f"{OUT}/environment.txt", "w").write("\n".join(env)); print("\n".join(env))
print("\nSaved files:", sorted(os.listdir(OUT)))
