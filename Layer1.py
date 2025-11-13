from Bio.Seq import Seq
from Bio import motifs
import numpy as np
from itertools import product
from collections import Counter
from scipy.stats import entropy
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report
import random
import os
import shap

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

sequences = []
with open("B_Enhancer.txt", "r") as file:
    seq = ""
    for line in file:
        line = line.strip()
        if line.startswith(">"):
            if seq:
                sequences.append(seq)
                seq = ""
        else:
            seq += line
    if seq:
        sequences.append(seq)

labels = [1 for i in range(len(sequences))]
with open("B_NonEnhancer.txt", "r") as file:
    seq = ""
    for line in file:
        line = line.strip()
        if line.startswith(">"):
            if seq:
                sequences.append(seq)
                seq = ""
        else:
            seq += line
    if seq:
        sequences.append(seq)
labels = labels + [0 for i in range(len(sequences) - len(labels))]

new_sequences = []
with open("I_Enhancer.txt", "r") as file:
    seq = ""
    for line in file:
        line = line.strip()
        if line.startswith(">"):
            if seq:
                new_sequences.append(seq)
                seq = ""
        else:
            seq += line
    if seq:
        new_sequences.append(seq)


new_labels = [1 for i in range(len(new_sequences))]


with open("I_NonEnhancer.txt", "r") as file:
    seq = ""
    for line in file:
        line = line.strip()
        if line.startswith(">"):
            if seq:
                new_sequences.append(seq)
                seq = ""
        else:
            seq += line
    if seq:
        new_sequences.append(seq)

new_labels = new_labels + [0 for i in range(len(new_sequences) - len(new_labels))]


def load_dna2vec(path, k=3):
    embedding_dict = {}
    with open(path, 'r') as f:
        for line in f:
            values = line.strip().split()
            kmer = values[0]
            if len(kmer) == k:
                vector = np.array(values[1:], dtype=np.float32)
                embedding_dict[kmer] = vector
    return embedding_dict

def kmer_tokenize_str(sequences, k=3):
    kmer_seqs = []
    for seq in sequences:
        seq = seq.upper()
        kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
        kmer_seqs.append(kmers)
    return kmer_seqs

def build_kmer_index(kmer_seqs):
    kmer_set = set(k for seq in kmer_seqs for k in seq)
    kmer_to_idx = {k: i+1 for i, k in enumerate(sorted(kmer_set))}
    return kmer_to_idx

def encode_kmers(kmer_seqs, kmer_to_idx):
    encoded = np.zeros((len(kmer_seqs), len(kmer_seqs[0])), dtype=np.int32)
    for i, seq in enumerate(kmer_seqs):
        for j, kmer in enumerate(seq):
            encoded[i, j] = kmer_to_idx.get(kmer, 0)
    return encoded

def build_embedding_matrix(kmer_to_idx, embedding_dict, embedding_dim):
    vocab_size = len(kmer_to_idx) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for kmer, idx in kmer_to_idx.items():
        if kmer in embedding_dict:
            embedding_matrix[idx] = embedding_dict[kmer]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.1, size=(embedding_dim,))
    return embedding_matrix


def seqs_to_embedded_vectors(encoded_seqs, embedding_matrix):
    vectors = []
    for seq in encoded_seqs:
        emb = embedding_matrix[seq]
        avg_emb = np.mean(emb, axis=0)
        vectors.append(avg_emb)
    return np.array(vectors)

k = 4
embedding_dim = 100
dna2vec_path = '/content/drive/MyDrive/dna2vec-20251027-1738-k3to5-100d-2c-0Mbp-sliding-338.w2v'
kmer_seqs = kmer_tokenize_str(sequences, k=k)
kmer_to_idx = build_kmer_index(kmer_seqs)
embedding_dict = load_dna2vec(dna2vec_path, k=k)
embedding_matrix = build_embedding_matrix(kmer_to_idx, embedding_dict, embedding_dim)
encoded_seqs = encode_kmers(kmer_seqs, kmer_to_idx)
kmer_seqs_new = kmer_tokenize_str(new_sequences, k=k)
encoded_seqs_new = encode_kmers(kmer_seqs_new, kmer_to_idx)
X_train = seqs_to_embedded_vectors(encoded_seqs, embedding_matrix)
X_test = seqs_to_embedded_vectors(encoded_seqs_new, embedding_matrix)

def performance(labelArr, predictArr):
    TN, FP, FN, TP = metrics.confusion_matrix(labelArr, predictArr).ravel()
    ACC = metrics.accuracy_score(labelArr, predictArr)
    SN = metrics.recall_score(labelArr, predictArr)
    SP = TN/(FP + TN)
    MCC= matthews_corrcoef(labelArr, predictArr)
    return ACC,SN,SP,MCC

# def Independence_test(Data12, Label12, Data34, Label34, param_grid):
#     estimator = XGBClassifier(eval_metric='logloss')
#     grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
#     grid_search.fit(Data12, Label12)
#     best_estimator = grid_search.best_estimator_
#     predictArr = best_estimator.predict(Data34)
#     acc, sens, spec, mcc = performance(Label34, predictArr)
#     return acc, sens, spec, mcc

def Independence_test(Data12, Label12, Data34, Label34, params):
    model = XGBClassifier(eval_metric='logloss',random_state=42, **params)
    model.fit(Data12, Label12)
    predictArr = model.predict(Data34)
    acc, sens, spec, mcc = performance(Label34, predictArr)
    importances = model.feature_importances_
    return acc, sens, spec, mcc,importances

def compute_gc_content(seq):
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) if len(seq) > 0 else 0

def compute_sequence_entropy(seq, k=3):
    seq = seq.upper()
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    kmer_counts = Counter(kmers)
    probs = [count / len(kmers) for count in kmer_counts.values()]
    return entropy(probs)

def extract_stat_features(seqs, k=3):
    features = []
    for seq in seqs:
        gc = compute_gc_content(seq)
        ent = compute_sequence_entropy(seq, k=k)
        features.append([gc, ent])
    return np.array(features)


def make_pwm_from_consensus(consensus):
    instances = [Seq(consensus)]
    m = motifs.create(instances)
    pwm = m.counts.normalize(pseudocounts=0.1)
    return pwm

def pwm_score_features(sequences, motif_list):
    pwm_logodds_list = []
    for motif in motif_list:
        pwm = make_pwm_from_consensus(motif)
        log_odds = pwm.log_odds()
        pwm_logodds_list.append(log_odds)

    features = []
    for seq in sequences:
        seq = Seq(seq.upper())
        seq_scores = []
        for log_odds in pwm_logodds_list:
            scores = [score for _, score in log_odds.search(seq)]
            max_score = max(scores) if scores else 0.0
            seq_scores.append(max_score)
        features.append(seq_scores)
    return np.array(features)


def get_all_kmers(k):
    return [''.join(p) for p in product('ACGT', repeat=k)]

def kmer_frequency_features(sequences, k=3):
    all_kmers = get_all_kmers(k)
    features = []
    for seq in sequences:
        seq = seq.upper()
        kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
        kmer_counts = Counter(kmers)
        total = sum(kmer_counts.values())
        freq_vector = [kmer_counts[kmer] / total if total > 0 else 0.0 for kmer in all_kmers]
        features.append(freq_vector)
    return np.array(features)


def dinuc_freq(sequences):
    dinucs = [a+b for a in 'ACGT' for b in 'ACGT']
    features = []
    for seq in sequences:
        seq = seq.upper()
        total = len(seq) - 1
        counts = Counter([seq[i:i+2] for i in range(total)])
        freq = [counts[d]/total if total > 0 else 0 for d in dinucs]
        features.append(freq)
    return np.array(features)


X_train_dinuc = dinuc_freq(sequences)
X_test_dinuc = dinuc_freq(new_sequences)
motifs_list = ['TATA','CCCTG','CCTGG','GCCTG','CCCAGG','CCCAGCC','TTGGGAG']
motif_features_train = pwm_score_features(sequences, motifs_list)
motif_features_test = pwm_score_features(new_sequences, motifs_list)
stat_features_train = extract_stat_features(sequences, k=3)
stat_features_test = extract_stat_features(new_sequences, k=3)
kmer_freq_train = kmer_frequency_features(sequences, k=3)
kmer_freq_test = kmer_frequency_features(new_sequences, k=3)


X_train = np.concatenate([ X_train, kmer_freq_train], axis=1)
X_test = np.concatenate([X_test, kmer_freq_test], axis=1)
X_train = np.concatenate([X_train, stat_features_train], axis=1)
X_test = np.concatenate([X_test, stat_features_test], axis=1)
X_train = np.concatenate([X_train, motif_features_train], axis=1)
X_test = np.concatenate([X_test, motif_features_test], axis=1)
X_train = np.concatenate([X_train, X_train_dinuc], axis=1)
X_test = np.concatenate([X_test, X_test_dinuc], axis=1)



param_grid = {
    'n_estimators': [150],
    'learning_rate': [0.2],
    'max_depth': [20],
    'min_child_weight': [4]
}
with open("accuracies_best.txt", "w") as f:
    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
              for min_child in param_grid['min_child_weight']:
                      acc, sens, spec, mcc,importances = Independence_test(X_train, labels, X_test, new_labels,
                                                              {'n_estimators': n_estimators,
                                                              'learning_rate': learning_rate,
                                                              'max_depth': max_depth,
                                                               'min_child_weight': min_child})
                      print("n_estimators:", n_estimators, "learning_rate:", learning_rate, "max_depth:", max_depth, "min_child_weight",min_child)
                      print("acc:", acc,"sen",sens,"spec",spec,"mcc",mcc)
                      f.write(f"n_estimators: {n_estimators}, learning_rate: {learning_rate}, max_depth: {max_depth}, acc: {acc}\n")
                      f.write(f"sens: {sens}, spec: {spec}, mcc: {mcc}\n")
                      f.write('------------------------------------------\n')

f.close()
