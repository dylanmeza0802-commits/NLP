import sys, json, time
sys.path.insert(0, "/home/claude/proyecto/src")
from utils import *
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, f1_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

set_seed()
RES = "/home/claude/proyecto/results"
FIG = "/home/claude/proyecto/figures"

train = pd.read_csv(f"{RES}/mc_train_clean.csv")
devel = pd.read_csv(f"{RES}/mc_devel_clean.csv")
test = pd.read_csv(f"{RES}/mc_test_clean.csv")

X_train, y_train = train["TEXT_CLEAN"].fillna(""), train["CLASS"]
X_devel, y_devel = devel["TEXT_CLEAN"].fillna(""), devel["CLASS"]
X_test, y_test = test["TEXT_CLEAN"].fillna(""), test["CLASS"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = {}

# ---------------- Logistic Regression ----------------
pipe_lr = Pipeline = None
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
])

param_grid_lr = {
    "tfidf__max_features": [20000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [2],
    "tfidf__sublinear_tf": [True],
    "clf__C": [0.1, 1, 5],
}

t0 = time.time()
gs_lr = GridSearchCV(pipe_lr, param_grid_lr, scoring="f1_macro", cv=skf, n_jobs=-1, verbose=1)
gs_lr.fit(X_train, y_train)
t_lr = time.time() - t0

results["logreg"] = {
    "best_params": gs_lr.best_params_,
    "best_cv_f1_macro": gs_lr.best_score_,
    "train_time_sec": t_lr,
}

# ---------------- Linear SVM ----------------
pipe_svm = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC(class_weight="balanced", random_state=SEED, max_iter=5000))
])
param_grid_svm = {
    "tfidf__max_features": [20000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [2],
    "tfidf__sublinear_tf": [True],
    "clf__C": [0.1, 1, 5],
}
t0 = time.time()
gs_svm = GridSearchCV(pipe_svm, param_grid_svm, scoring="f1_macro", cv=skf, n_jobs=-1, verbose=1)
gs_svm.fit(X_train, y_train)
t_svm = time.time() - t0

results["linear_svm"] = {
    "best_params": gs_svm.best_params_,
    "best_cv_f1_macro": gs_svm.best_score_,
    "train_time_sec": t_svm,
}

# ---------------- Char n-grams experiment (adicional) ----------------
pipe_char = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1, random_state=SEED))
])
t0 = time.time()
pipe_char.fit(X_train, y_train)
t_char = time.time() - t0

# ---------------- Evaluación en DEVEL para elegir el mejor modelo ----------------
def evaluate(model, X, y, name):
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro"),
        "f1_micro": f1_score(y, preds, average="micro"),
        "precision_macro": precision_score(y, preds, average="macro"),
        "recall_macro": recall_score(y, preds, average="macro"),
    }, preds

eval_devel = {}
eval_devel["logreg"], _ = evaluate(gs_lr.best_estimator_, X_devel, y_devel, "logreg")
eval_devel["linear_svm"], _ = evaluate(gs_svm.best_estimator_, X_devel, y_devel, "linear_svm")
eval_devel["char_ngrams_logreg"], _ = evaluate(pipe_char, X_devel, y_devel, "char")

results["evaluacion_devel"] = eval_devel

# Elegir el mejor por F1-macro en devel
best_name = max(eval_devel, key=lambda k: eval_devel[k]["f1_macro"])
best_model = {"logreg": gs_lr.best_estimator_, "linear_svm": gs_svm.best_estimator_, "char_ngrams_logreg": pipe_char}[best_name]
results["mejor_modelo_baseline"] = best_name

# ---------------- Evaluación final en TEST (solo con el mejor modelo) ----------------
test_metrics, test_preds = evaluate(best_model, X_test, y_test, best_name)
results["evaluacion_test_mejor_modelo"] = test_metrics

report_dict = classification_report(y_test, test_preds, target_names=[LABEL_NAMES_MULTICLASS[i] for i in range(4)], output_dict=True)
results["metricas_por_clase_test"] = report_dict

cm = confusion_matrix(y_test, test_preds)
results["matriz_confusion_test"] = cm.tolist()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[LABEL_NAMES_MULTICLASS[i] for i in range(4)],
            yticklabels=[LABEL_NAMES_MULTICLASS[i] for i in range(4)], ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusión - Test set\nBaseline: {best_name} (F1-macro={test_metrics['f1_macro']:.3f})")
plt.tight_layout()
plt.savefig(f"{FIG}/07_confusion_matrix_baseline_multiclass.png", dpi=130)
plt.close()

# Guardar el mejor modelo y las predicciones de test para análisis de errores (Fase 7)
joblib.dump(best_model, f"{RES}/best_baseline_multiclass.joblib")
test_out = test.copy()
test_out["pred"] = test_preds
test_out.to_csv(f"{RES}/mc_test_predictions_baseline.csv", index=False)

with open(f"{RES}/phase3_baseline_multiclass_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print("FASE 3 COMPLETADA")
print(json.dumps({
    "cv_f1_macro_logreg": results["logreg"]["best_cv_f1_macro"],
    "cv_f1_macro_svm": results["linear_svm"]["best_cv_f1_macro"],
    "evaluacion_devel": eval_devel,
    "mejor_modelo": best_name,
    "test_metrics": test_metrics,
}, ensure_ascii=False, indent=2))
