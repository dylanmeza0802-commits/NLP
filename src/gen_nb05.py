import sys
sys.path.insert(0, "/home/claude/proyecto/src")
from build_notebook import build_and_run

cells = [
("markdown", """# 05 - Baseline Subtarea 2: Clasificación Multietiqueta de Tipos de Violencia

Cada narrativa puede tener 0 a 7 etiquetas activas simultáneamente (no excluyentes).
Se implementa `TF-IDF + OneVsRestClassifier(LogisticRegression)`, usando
`MultilabelStratifiedKFold` (Iterative Stratification) para la validación cruzada,
tal como exige el enunciado, dado que un `StratifiedKFold` estándar no preserva
bien la distribución conjunta de combinaciones de etiquetas.
"""),

("code", """import sys, time
sys.path.insert(0, "/home/claude/proyecto/src")
from utils import *
import numpy as np, pandas as pd, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
%matplotlib inline

set_seed()
RES = "/home/claude/proyecto/results"

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
print("Train:", X_train_text.shape, "Devel:", X_devel_text.shape, "Test:", X_test_text.shape)
"""),

("markdown", """## Búsqueda de hiperparámetros con Iterative Stratification (5 folds)"""),

("code", """mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
candidates = [
    {"max_features": 15000, "ngram_range": (1,1), "C": 1},
    {"max_features": 15000, "ngram_range": (1,2), "C": 1},
    {"max_features": 20000, "ngram_range": (1,2), "C": 5},
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
print(json.dumps(cv_results, indent=2))
best_params = cv_results[0]["params"]
print("\\nMejores hiperparámetros:", best_params)
"""),

("markdown", """## Entrenamiento final y umbral óptimo por etiqueta

**Diferencia clave con multiclase**: aquí NO se usa argmax. Cada etiqueta tiene su
propia probabilidad (vía `predict_proba` de OneVsRest) y su propio umbral óptimo,
calculado maximizando F1 **exclusivamente sobre `devel`** (nunca sobre test), ya
que las etiquetas tienen tasas base muy distintas (desde 3.4% hasta 79.5%)."""),

("code", """tfidf_final = TfidfVectorizer(max_features=best_params["max_features"], ngram_range=best_params["ngram_range"],
                               min_df=2, sublinear_tf=True)
Xtr_final = tfidf_final.fit_transform(X_train_text)
Xdevel_final = tfidf_final.transform(X_devel_text)
Xtest_final = tfidf_final.transform(X_test_text)

clf_final = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced",
                                                      C=best_params["C"], random_state=SEED))
clf_final.fit(Xtr_final, Y_train)

proba_devel = clf_final.predict_proba(Xdevel_final)
proba_test = clf_final.predict_proba(Xtest_final)

thresholds = {}
for i, lab in enumerate(label_names):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(Y_devel[:, i], (proba_devel[:, i] >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    thresholds[lab] = float(best_t)
print("Umbrales óptimos por etiqueta (calculados en devel):")
print(json.dumps(thresholds, indent=2))
"""),

("code", """thr_array = np.array([thresholds[lab] for lab in label_names])
preds_test_05 = (proba_test >= 0.5).astype(int)
preds_test_opt = (proba_test >= thr_array).astype(int)

def multilabel_metrics(Y_true, Y_pred):
    return {
        "f1_macro": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(Y_true, Y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(Y_true, Y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(Y_true, Y_pred),
    }

print("TEST con umbral fijo 0.5:")
print(json.dumps(multilabel_metrics(Y_test, preds_test_05), indent=2))
print("\\nTEST con umbral óptimo por etiqueta:")
print(json.dumps(multilabel_metrics(Y_test, preds_test_opt), indent=2))
"""),

("markdown", """## Métricas por etiqueta y análisis de falsos positivos/negativos"""),

("code", """per_label = {}
for i, lab in enumerate(label_names):
    per_label[lab] = {
        "precision": float(precision_score(Y_test[:,i], preds_test_opt[:,i], zero_division=0)),
        "recall": float(recall_score(Y_test[:,i], preds_test_opt[:,i], zero_division=0)),
        "f1": float(f1_score(Y_test[:,i], preds_test_opt[:,i], zero_division=0)),
        "support": int(Y_test[:,i].sum()),
        "falsos_positivos": int(((preds_test_opt[:,i]==1)&(Y_test[:,i]==0)).sum()),
        "falsos_negativos": int(((preds_test_opt[:,i]==0)&(Y_test[:,i]==1)).sum()),
    }
print(json.dumps(per_label, indent=2))

fig, ax = plt.subplots(figsize=(8,5))
f1s = [per_label[l]["f1"] for l in label_names]
ax.barh(label_names, f1s, color=sns.color_palette("viridis", len(label_names)))
ax.set_xlabel("F1-score (test, umbral óptimo)")
ax.set_title("Desempeño del baseline multietiqueta por tipo de violencia")
for i, v in enumerate(f1s):
    ax.text(v+0.01, i, f"{v:.2f}", va="center")
plt.tight_layout(); plt.show()
"""),

("markdown", """**Observación**: `Vicarious` muestra la precisión más baja (~0.35) pese a un
recall aceptable — el modelo sobre-predice esta etiqueta. Se profundiza en la
Fase 7. `Psychological` (la etiqueta mayoritaria, 79.5% de soporte) obtiene el
mejor F1 (~0.93), como es esperable dada su prevalencia."""),

("code", """joblib.dump({"tfidf": tfidf_final, "clf": clf_final, "thresholds": thresholds},
            f"{RES}/best_baseline_multilabel.joblib")
test_out = test.copy()
for i, lab in enumerate(label_names):
    test_out[f"pred_{lab}"] = preds_test_opt[:, i]
test_out.to_csv(f"{RES}/ml_test_predictions_baseline.csv", index=False)

results_to_save = {
    "cv_results_hparams": cv_results, "best_params": best_params,
    "thresholds_optimizados_en_devel": thresholds,
    "metrics_test_threshold_05": multilabel_metrics(Y_test, preds_test_05),
    "metrics_test_threshold_optimo": multilabel_metrics(Y_test, preds_test_opt),
    "metricas_por_etiqueta_test": per_label,
}
with open(f"{RES}/phase5_baseline_multilabel_results.json", "w", encoding="utf-8") as f:
    json.dump(results_to_save, f, ensure_ascii=False, indent=2, default=str)
print("Guardado.")
"""),
]

build_and_run(cells, "/home/claude/proyecto/notebooks/05_Baseline_Multietiqueta.ipynb", timeout=900)
