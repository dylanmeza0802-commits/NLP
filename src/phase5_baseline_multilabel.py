import sys, json, time
sys.path.insert(0, "/home/claude/proyecto/src")
from utils import *
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score, hamming_loss,
                              classification_report)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

set_seed()
RES = "/home/claude/proyecto/results"
FIG = "/home/claude/proyecto/figures"

train = pd.read_csv(f"{RES}/ml_train_clean.csv")
devel = pd.read_csv(f"{RES}/ml_devel_clean.csv")
test = pd.read_csv(f"{RES}/ml_test_clean.csv")

X_train_text = train["TEXT_CLEAN"].fillna("")
X_devel_text = devel["TEXT_CLEAN"].fillna("")
X_test_text = test["TEXT_CLEAN"].fillna("")

Y_train = train[MULTILABEL_COLS].values
Y_devel = devel[MULTILABEL_COLS].values
Y_test = test[MULTILABEL_COLS].values

label_names = [LABEL_NAMES_MULTILABEL[c] for c in MULTILABEL_COLS]

# ---- Validación cruzada con Iterative Stratification (para explorar hiperparámetros de TF-IDF) ----
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

candidates = [
    {"max_features": 15000, "ngram_range": (1, 1), "C": 1},
    {"max_features": 15000, "ngram_range": (1, 2), "C": 1},
    {"max_features": 20000, "ngram_range": (1, 2), "C": 5},
]

cv_results = []
for cand in candidates:
    fold_f1s = []
    for tr_idx, val_idx in mskf.split(X_train_text, Y_train):
        tfidf = TfidfVectorizer(max_features=cand["max_features"], ngram_range=cand["ngram_range"],
                                 min_df=2, sublinear_tf=True)
        Xtr = tfidf.fit_transform(X_train_text.iloc[tr_idx])
        Xval = tfidf.transform(X_train_text.iloc[val_idx])
        clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced",
                                                        C=cand["C"], random_state=SEED))
        clf.fit(Xtr, Y_train[tr_idx])
        preds = clf.predict(Xval)
        fold_f1s.append(f1_score(Y_train[val_idx], preds, average="macro", zero_division=0))
    cv_results.append({"params": cand, "mean_f1_macro": float(np.mean(fold_f1s)), "std": float(np.std(fold_f1s))})

cv_results.sort(key=lambda r: -r["mean_f1_macro"])
best_params = cv_results[0]["params"]

# ---- Entrenar modelo final con mejores hiperparámetros sobre todo el train ----
tfidf_final = TfidfVectorizer(max_features=best_params["max_features"], ngram_range=best_params["ngram_range"],
                               min_df=2, sublinear_tf=True)
Xtr_final = tfidf_final.fit_transform(X_train_text)
Xdevel_final = tfidf_final.transform(X_devel_text)
Xtest_final = tfidf_final.transform(X_test_text)

clf_final = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced",
                                                      C=best_params["C"], random_state=SEED))
clf_final.fit(Xtr_final, Y_train)

# Probabilidades para umbral óptimo por etiqueta (usando SOLO devel, nunca test)
proba_devel = clf_final.predict_proba(Xdevel_final)
proba_test = clf_final.predict_proba(Xtest_final)

# Umbral óptimo por etiqueta maximizando F1 en devel
thresholds = {}
for i, lab in enumerate(label_names):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.05):
        preds_i = (proba_devel[:, i] >= t).astype(int)
        f1 = f1_score(Y_devel[:, i], preds_i, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    thresholds[lab] = float(best_t)

thr_array = np.array([thresholds[lab] for lab in label_names])
preds_devel_opt = (proba_devel >= thr_array).astype(int)
preds_test_opt = (proba_test >= thr_array).astype(int)

# También con umbral fijo 0.5 para comparar
preds_test_05 = (proba_test >= 0.5).astype(int)

def multilabel_metrics(Y_true, Y_pred):
    return {
        "f1_macro": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(Y_true, Y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(Y_true, Y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(Y_true, Y_pred),
    }

results = {
    "cv_results_hparams": cv_results,
    "best_params": best_params,
    "thresholds_optimizados_en_devel": thresholds,
    "metrics_devel_threshold_optimo": multilabel_metrics(Y_devel, preds_devel_opt),
    "metrics_test_threshold_05": multilabel_metrics(Y_test, preds_test_05),
    "metrics_test_threshold_optimo": multilabel_metrics(Y_test, preds_test_opt),
}

# Métricas por etiqueta en test (con umbral óptimo)
per_label = {}
for i, lab in enumerate(label_names):
    per_label[lab] = {
        "precision": float(precision_score(Y_test[:, i], preds_test_opt[:, i], zero_division=0)),
        "recall": float(recall_score(Y_test[:, i], preds_test_opt[:, i], zero_division=0)),
        "f1": float(f1_score(Y_test[:, i], preds_test_opt[:, i], zero_division=0)),
        "support": int(Y_test[:, i].sum()),
        "threshold": thresholds[lab],
    }
results["metricas_por_etiqueta_test"] = per_label

# Análisis rápido de falsos positivos / negativos por etiqueta
fp_fn = {}
for i, lab in enumerate(label_names):
    fp = int(((preds_test_opt[:, i] == 1) & (Y_test[:, i] == 0)).sum())
    fn = int(((preds_test_opt[:, i] == 0) & (Y_test[:, i] == 1)).sum())
    fp_fn[lab] = {"falsos_positivos": fp, "falsos_negativos": fn}
results["falsos_positivos_negativos_por_etiqueta"] = fp_fn

# Gráfico F1 por etiqueta
fig, ax = plt.subplots(figsize=(8, 5))
f1s = [per_label[l]["f1"] for l in label_names]
sns.barplot(x=f1s, y=label_names, palette="viridis", ax=ax)
ax.set_xlabel("F1-score (test, umbral óptimo)")
ax.set_title("Desempeño del baseline multietiqueta por tipo de violencia")
for i, v in enumerate(f1s):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center")
plt.tight_layout()
plt.savefig(f"{FIG}/08_f1_por_etiqueta_baseline.png", dpi=130)
plt.close()

joblib.dump({"tfidf": tfidf_final, "clf": clf_final, "thresholds": thresholds}, f"{RES}/best_baseline_multilabel.joblib")
test_out = test.copy()
for i, lab in enumerate(label_names):
    test_out[f"pred_{lab}"] = preds_test_opt[:, i]
test_out.to_csv(f"{RES}/ml_test_predictions_baseline.csv", index=False)

with open(f"{RES}/phase5_baseline_multilabel_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print("FASE 5 COMPLETADA")
print(json.dumps({
    "cv_results": cv_results,
    "metrics_test_threshold_05": results["metrics_test_threshold_05"],
    "metrics_test_threshold_optimo": results["metrics_test_threshold_optimo"],
    "per_label": per_label,
}, ensure_ascii=False, indent=2))
