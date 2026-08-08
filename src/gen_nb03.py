import sys
sys.path.insert(0, "/home/claude/proyecto/src")
from build_notebook import build_and_run

cells = [
("markdown", """# 03 - Baseline Subtarea 1: Clasificación Multiclase de Severidad

Se implementan y comparan **TF-IDF + Logistic Regression** y **TF-IDF + Linear SVM**,
con optimización de hiperparámetros mediante `GridSearchCV` y `StratifiedKFold` (5
folds) sobre el conjunto de entrenamiento. El conjunto de test **no se toca** durante
la selección de modelo/hiperparámetros; solo se usa al final, una única vez, sobre
el mejor modelo elegido por desempeño en `devel`.
"""),

("code", """import sys, time
sys.path.insert(0, "/home/claude/proyecto/src")
from utils import *
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, f1_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
%matplotlib inline

set_seed()
RES = "/home/claude/proyecto/results"

train = pd.read_csv(f"{RES}/mc_train_clean.csv")
devel = pd.read_csv(f"{RES}/mc_devel_clean.csv")
test = pd.read_csv(f"{RES}/mc_test_clean.csv")

X_train, y_train = train["TEXT_CLEAN"].fillna(""), train["CLASS"]
X_devel, y_devel = devel["TEXT_CLEAN"].fillna(""), devel["CLASS"]
X_test, y_test = test["TEXT_CLEAN"].fillna(""), test["CLASS"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("Train:", X_train.shape, " Devel:", X_devel.shape, " Test:", X_test.shape)
"""),

("markdown", """## Grid de hiperparámetros

Se exploran: `max_features`, unigrama vs. unigrama+bigrama, y la fuerza de
regularización `C`, usando `class_weight="balanced"` para mitigar el desbalance
(ratio 8.6:1) detectado en la Fase 1."""),

("code", """pipe_lr = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
])
param_grid_lr = {
    "tfidf__max_features": [20000],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [2],
    "tfidf__sublinear_tf": [True],
    "clf__C": [0.1, 1, 5],
}
gs_lr = GridSearchCV(pipe_lr, param_grid_lr, scoring="f1_macro", cv=skf, n_jobs=-1)
gs_lr.fit(X_train, y_train)
print("Mejor combinación (LogReg):", gs_lr.best_params_)
print("F1-macro (CV, train):", gs_lr.best_score_)
"""),

("code", """pipe_svm = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC(class_weight="balanced", random_state=SEED, max_iter=5000))
])
param_grid_svm = {
    "tfidf__max_features": [20000],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [2],
    "tfidf__sublinear_tf": [True],
    "clf__C": [0.1, 1, 5],
}
gs_svm = GridSearchCV(pipe_svm, param_grid_svm, scoring="f1_macro", cv=skf, n_jobs=-1)
gs_svm.fit(X_train, y_train)
print("Mejor combinación (SVM):", gs_svm.best_params_)
print("F1-macro (CV, train):", gs_svm.best_score_)
"""),

("markdown", """## Experimento adicional: n-gramas de caracteres

Los n-gramas de caracteres (3-5) son robustos a los errores ortográficos
detectados en el EDA (ej. "FAMILAIR" y "FAMILIAR" comparten muchos trigramas)."""),

("code", """pipe_char = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=30000, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1, random_state=SEED))
])
pipe_char.fit(X_train, y_train)
print("Entrenado (char n-grams).")
"""),

("markdown", """## Selección del mejor modelo usando `devel` (no `test`)"""),

("code", """def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro"),
        "f1_micro": f1_score(y, preds, average="micro"),
        "precision_macro": precision_score(y, preds, average="macro"),
        "recall_macro": recall_score(y, preds, average="macro"),
    }, preds

eval_devel = {}
eval_devel["logreg"], _ = evaluate(gs_lr.best_estimator_, X_devel, y_devel)
eval_devel["linear_svm"], _ = evaluate(gs_svm.best_estimator_, X_devel, y_devel)
eval_devel["char_ngrams_logreg"], _ = evaluate(pipe_char, X_devel, y_devel)

import json
print(json.dumps(eval_devel, indent=2))

best_name = max(eval_devel, key=lambda k: eval_devel[k]["f1_macro"])
best_model = {"logreg": gs_lr.best_estimator_, "linear_svm": gs_svm.best_estimator_,
              "char_ngrams_logreg": pipe_char}[best_name]
print("\\nMejor modelo baseline (por F1-macro en devel):", best_name)
"""),

("markdown", """**Observación importante**: el F1-macro de validación cruzada sobre `train`
(~0.65-0.67) es notablemente más alto que el obtenido en `devel` (~0.47-0.48).
Esta brecha sugiere que la distribución de `train` y `devel/test` difiere
ligeramente (posible variación temporal o de fuente en la recolección de datos),
un patrón similar al detectado en otros proyectos con este tipo de datos
administrativos reales. Se profundiza en la Fase 7 (Análisis de Errores)."""),

("markdown", """## Evaluación final en TEST (una sola vez, con el mejor modelo)"""),

("code", """test_metrics, test_preds = evaluate(best_model, X_test, y_test)
print("Métricas finales en TEST:")
print(json.dumps(test_metrics, indent=2))

print("\\nReporte por clase:")
print(classification_report(y_test, test_preds, target_names=[LABEL_NAMES_MULTICLASS[i] for i in range(4)]))
"""),

("code", """cm = confusion_matrix(y_test, test_preds)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[LABEL_NAMES_MULTICLASS[i] for i in range(4)],
            yticklabels=[LABEL_NAMES_MULTICLASS[i] for i in range(4)], ax=ax)
ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusión - Test\\nBaseline: {best_name} (F1-macro={test_metrics['f1_macro']:.3f})")
plt.tight_layout(); plt.show()
"""),

("code", """# Guardar modelo y predicciones para las Fases 7 y 8
joblib.dump(best_model, f"{RES}/best_baseline_multiclass.joblib")
test_out = test.copy()
test_out["pred"] = test_preds
test_out.to_csv(f"{RES}/mc_test_predictions_baseline.csv", index=False)

results_to_save = {
    "logreg": {"best_params": gs_lr.best_params_, "best_cv_f1_macro": gs_lr.best_score_},
    "linear_svm": {"best_params": gs_svm.best_params_, "best_cv_f1_macro": gs_svm.best_score_},
    "evaluacion_devel": eval_devel,
    "mejor_modelo_baseline": best_name,
    "evaluacion_test_mejor_modelo": test_metrics,
    "matriz_confusion_test": cm.tolist(),
}
with open(f"{RES}/phase3_baseline_multiclass_results.json", "w", encoding="utf-8") as f:
    json.dump(results_to_save, f, ensure_ascii=False, indent=2, default=str)
print("Guardado.")
"""),

("markdown", """## Interpretabilidad: palabras más influyentes por clase"""),

("code", """if best_name != "char_ngrams_logreg":
    tfidf = best_model.named_steps["tfidf"]
    clf = best_model.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())
    if hasattr(clf, "coef_"):
        fig, axes = plt.subplots(1, 4, figsize=(20,5))
        for i, ax in enumerate(axes):
            coefs = clf.coef_[i]
            top_idx = np.argsort(coefs)[-10:]
            ax.barh(feature_names[top_idx], coefs[top_idx], color="teal")
            ax.set_title(f"Top features - {LABEL_NAMES_MULTICLASS[i]}")
        plt.tight_layout(); plt.show()
"""),

("markdown", """## Resumen Fase 3

| Modelo | F1-macro CV (train) | F1-macro (devel) | F1-macro (test) |
|---|---|---|---|
| Ver celdas de salida arriba para los valores exactos obtenidos en esta ejecución. |

El mejor baseline seleccionado se guarda en `results/best_baseline_multiclass.joblib`
y sus predicciones en `results/mc_test_predictions_baseline.csv`, insumos para la
Fase 7 (análisis de errores) y la tabla comparativa Baseline vs. Transformer del
informe final (Fase 8).
"""),
]

build_and_run(cells, "/home/claude/proyecto/notebooks/03_Baseline_Multiclase.ipynb", timeout=900)
