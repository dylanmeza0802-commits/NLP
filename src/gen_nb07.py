import sys
sys.path.insert(0, "/home/claude/proyecto/src")
from build_notebook import build_and_run

cells = [
("markdown", """# 07 - Evaluación Comparativa y Análisis de Errores

Este notebook consolida las métricas de los modelos **Baseline** entrenados
(Fases 3 y 5) y realiza el análisis cualitativo de errores exigido por la
rúbrica. **Los resultados del Transformer (BETO) se incorporarán aquí una vez
ejecutados en Colab** (Fases 4 y 6) — no se completan con valores inventados.
"""),

("code", """import sys, json
sys.path.insert(0, "/home/claude/proyecto/src")
from utils import *
import pandas as pd
import numpy as np
import os

RES = "/home/claude/proyecto/results"

with open(f"{RES}/phase3_baseline_multiclass_results.json", encoding="utf-8") as f:
    mc_results = json.load(f)
with open(f"{RES}/phase5_baseline_multilabel_results.json", encoding="utf-8") as f:
    ml_results = json.load(f)

beto_mc_path = f"{RES}/phase4_beto_multiclass_results.json"
beto_ml_path = f"{RES}/phase6_beto_multilabel_results.json"
beto_mc_available = os.path.exists(beto_mc_path)
beto_ml_available = os.path.exists(beto_ml_path)
print("Resultados BETO multiclase disponibles:", beto_mc_available)
print("Resultados BETO multietiqueta disponibles:", beto_ml_available)
"""),

("markdown", """## Tabla comparativa — Subtarea 1 (Multiclase)"""),

("code", """rows = [{
    "Modelo": f"Baseline ({mc_results['mejor_modelo_baseline']})",
    "Accuracy": mc_results["evaluacion_test_mejor_modelo"]["accuracy"],
    "F1-Macro": mc_results["evaluacion_test_mejor_modelo"]["f1_macro"],
    "F1-Micro": mc_results["evaluacion_test_mejor_modelo"]["f1_micro"],
    "Precision-Macro": mc_results["evaluacion_test_mejor_modelo"]["precision_macro"],
    "Recall-Macro": mc_results["evaluacion_test_mejor_modelo"]["recall_macro"],
}]
if beto_mc_available:
    with open(beto_mc_path, encoding="utf-8") as f:
        beto_mc = json.load(f)
    tm = beto_mc["test_metrics"]
    rows.append({"Modelo": "BETO (fine-tuned)", "Accuracy": tm["accuracy"], "F1-Macro": tm["f1_macro"],
                 "F1-Micro": tm["f1_micro"], "Precision-Macro": tm["precision_macro"], "Recall-Macro": tm["recall_macro"]})
else:
    rows.append({"Modelo": "BETO (fine-tuned)", "Accuracy": "PENDIENTE (ejecutar Fase 4 en Colab)",
                 "F1-Macro": "-", "F1-Micro": "-", "Precision-Macro": "-", "Recall-Macro": "-"})

pd.DataFrame(rows).set_index("Modelo")
"""),

("markdown", """## Tabla comparativa — Subtarea 2 (Multietiqueta)"""),

("code", """rows2 = [{
    "Modelo": "Baseline (TF-IDF + OvR LogReg)",
    "F1-Macro": ml_results["metrics_test_threshold_optimo"]["f1_macro"],
    "F1-Micro": ml_results["metrics_test_threshold_optimo"]["f1_micro"],
    "Precision-Macro": ml_results["metrics_test_threshold_optimo"]["precision_macro"],
    "Recall-Macro": ml_results["metrics_test_threshold_optimo"]["recall_macro"],
    "Hamming Loss": ml_results["metrics_test_threshold_optimo"]["hamming_loss"],
}]
if beto_ml_available:
    with open(beto_ml_path, encoding="utf-8") as f:
        beto_ml = json.load(f)
    tm = beto_ml["test_metrics"]
    rows2.append({"Modelo": "BETO (fine-tuned)", "F1-Macro": tm["f1_macro"], "F1-Micro": tm["f1_micro"],
                  "Precision-Macro": tm["precision_macro"], "Recall-Macro": tm["recall_macro"],
                  "Hamming Loss": tm["hamming_loss"]})
else:
    rows2.append({"Modelo": "BETO (fine-tuned)", "F1-Macro": "PENDIENTE (ejecutar Fase 6 en Colab)",
                  "F1-Micro": "-", "Precision-Macro": "-", "Recall-Macro": "-", "Hamming Loss": "-"})

pd.DataFrame(rows2).set_index("Modelo")
"""),

("markdown", """## Análisis de errores críticos — Subtarea 1

Foco especial en `Severe -> Mild` (el error más peligroso: subestimar un caso
grave) y en confusiones entre clases adyacentes."""),

("code", """mc_pred = pd.read_csv(f"{RES}/mc_test_predictions_baseline.csv")

severe_to_mild = mc_pred[(mc_pred["CLASS"]==3) & (mc_pred["pred"]==0)]
print(f"Casos Severe clasificados como Mild: {len(severe_to_mild)} de {sum(mc_pred['CLASS']==3)} casos Severe")
print()
for _, row in severe_to_mild.head(3).iterrows():
    print(">", row["TEXT"][:250])
    print()
"""),

("code", """severe_wrong = mc_pred[(mc_pred["CLASS"]==3) & (mc_pred["pred"]!=3)]
dist = severe_wrong["pred"].value_counts().rename(index=LABEL_NAMES_MULTICLASS)
print("Distribución de errores cuando la clase real es Severe:")
print(dist)
print(f"\\nLa mayoría de los errores de Severe caen en 'High' (clase adyacente), no en 'Mild'.")
print("Esto es relativamente tranquilizador: el modelo rara vez comete el error más grave (Severe->Mild),")
print("aunque sí subestima consistentemente la severidad máxima.")
"""),

("code", """mc_pred["len_words"] = mc_pred["TEXT"].str.split().apply(len)
short = mc_pred[mc_pred["len_words"] <= 15]
short_wrong = short[short["CLASS"] != short["pred"]]
print(f"Narrativas cortas (<=15 palabras): {len(short)}")
print(f"De ellas, mal clasificadas: {len(short_wrong)} ({len(short_wrong)/len(short)*100:.1f}%)")
print(f"Tasa de error general del modelo: {(mc_pred['CLASS']!=mc_pred['pred']).mean()*100:.1f}%")
print("\\n-> Las narrativas cortas tienen una tasa de error sustancialmente mayor,")
print("   confirmando la hipótesis de la Fase 1: poca longitud = poca señal semántica.")
for _, row in short_wrong.head(3).iterrows():
    print("\\n>", row['TEXT'], "| Real:", LABEL_NAMES_MULTICLASS[row['CLASS']], "| Pred:", LABEL_NAMES_MULTICLASS[row['pred']])
"""),

("markdown", """## Análisis de errores — Subtarea 2 (Falsos negativos en etiquetas críticas)

Los falsos negativos en `Sexual`, `Physical` y `Vicarious` son los más graves desde
una perspectiva de protección de la víctima (no detectar un tipo de violencia
presente)."""),

("code", """ml_pred = pd.read_csv(f"{RES}/ml_test_predictions_baseline.csv")
label_names = [LABEL_NAMES_MULTILABEL[c] for c in MULTILABEL_COLS]

for lab in ["Sexual", "Physical", "Vicarious"]:
    true_col = MULTILABEL_COLS[label_names.index(lab)]
    pred_col = f"pred_{lab}"
    fn = ml_pred[(ml_pred[true_col]==1) & (ml_pred[pred_col]==0)]
    print(f"{lab}: {len(fn)} falsos negativos de {ml_pred[true_col].sum()} casos reales")
    if len(fn) > 0:
        print("  Ejemplo:", fn['Text'].iloc[0][:200])
    print()
"""),

("code", """vic_col = MULTILABEL_COLS[label_names.index("Vicarious")]
vic_fp = ml_pred[(ml_pred[vic_col]==0) & (ml_pred["pred_Vicarious"]==1)]
print(f"Vicarious: {len(vic_fp)} falsos positivos (el modelo sobre-predice esta etiqueta)")
for t in vic_fp["Text"].head(3):
    print(">", t[:200])
    print()
print("Hipótesis: 'Vicarious' probablemente se confunde con narrativas de custodia/hijos")
print("que NO implican uso instrumental de los hijos como forma de violencia (solo trámites legales).")
"""),

("markdown", """## Síntesis del análisis de errores

1. **Error crítico Severe->Mild**: ocurre en un número reducido de casos (ver
   celda superior); la mayoría de los errores de Severe caen en High (clase
   adyacente), no en el extremo opuesto — patrón menos peligroso pero aún relevante
   para la sección de Ética.
2. **Narrativas cortas** concentran una proporción de error notablemente mayor
   que el promedio general — limitación estructural del texto, no solo del modelo.
3. **Brecha CV vs. test** en la Subtarea 1 sugiere posible variación de
   distribución entre el momento de recolección de train y de devel/test —
   se recomienda como trabajo futuro un análisis temporal de los datos si existe
   metadato de fecha.
4. **Vicarious** es la etiqueta más difícil (falsos positivos con narrativas de
   custodia sin violencia instrumental hacia los hijos) — la etiqueta con menor
   soporte (11%) y mayor solapamiento conceptual con trámites legales rutinarios.

*Pendiente*: una vez completadas las Fases 4 y 6 (BETO) en Colab, se debe repetir
este análisis de errores con las predicciones del Transformer y comparar
cualitativamente si BETO resuelve o no estos mismos patrones de error.
"""),
]

build_and_run(cells, "/home/claude/proyecto/notebooks/07_Evaluacion_Analisis_Errores.ipynb", timeout=300)
