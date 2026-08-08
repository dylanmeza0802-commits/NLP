import sys, json
sys.path.insert(0, "/home/claude/proyecto/src")
from utils import *
import pandas as pd
import numpy as np

RES = "/home/claude/proyecto/results"

mc_pred = pd.read_csv(f"{RES}/mc_test_predictions_baseline.csv")
ml_pred = pd.read_csv(f"{RES}/ml_test_predictions_baseline.csv")

report = {}

# ---- Errores críticos: Severe(3) clasificado como Mild(0) ----
severe_to_mild = mc_pred[(mc_pred["CLASS"] == 3) & (mc_pred["pred"] == 0)]
report["n_errores_criticos_severe_a_mild"] = len(severe_to_mild)
report["ejemplos_severe_a_mild"] = severe_to_mild[["TEXT", "CLASS", "pred"]].head(5).to_dict("records")

# Severe mal clasificado en general
severe_wrong = mc_pred[(mc_pred["CLASS"] == 3) & (mc_pred["pred"] != 3)]
report["distribucion_errores_de_severe"] = severe_wrong["pred"].value_counts().rename(index=LABEL_NAMES_MULTICLASS).to_dict()
report["ejemplos_severe_mal_clasificado"] = severe_wrong[["TEXT", "CLASS", "pred"]].head(5).assign(
    pred_label=lambda d: d["pred"].map(LABEL_NAMES_MULTICLASS)
).to_dict("records")

# Narrativas cortas mal clasificadas
mc_pred["len_words"] = mc_pred["TEXT"].str.split().apply(len)
short_wrong = mc_pred[(mc_pred["len_words"] <= 15) & (mc_pred["CLASS"] != mc_pred["pred"])]
report["n_narrativas_cortas_mal_clasificadas"] = len(short_wrong)
report["pct_narrativas_cortas_mal_clasificadas"] = len(short_wrong) / max(1, len(mc_pred[mc_pred["len_words"] <= 15]))
report["ejemplos_cortas_mal_clasificadas"] = short_wrong[["TEXT", "CLASS", "pred"]].head(5).to_dict("records")

# Confusión Medium<->High (la más frecuente según matriz de confusión)
med_high_confusion = mc_pred[((mc_pred["CLASS"] == 1) & (mc_pred["pred"] == 2)) |
                              ((mc_pred["CLASS"] == 2) & (mc_pred["pred"] == 1))]
report["n_confusion_medium_high"] = len(med_high_confusion)
report["ejemplos_confusion_medium_high"] = med_high_confusion[["TEXT", "CLASS", "pred"]].head(4).to_dict("records")

# ---- Multilabel: falsos negativos en etiquetas críticas (Sexual, Physical, Vicarious) ----
label_names = [LABEL_NAMES_MULTILABEL[c] for c in MULTILABEL_COLS]
critical_labels = ["Sexual", "Physical", "Vicarious"]
ml_errors = {}
for lab in critical_labels:
    true_col = MULTILABEL_COLS[label_names.index(lab)]
    pred_col = f"pred_{lab}"
    fn = ml_pred[(ml_pred[true_col] == 1) & (ml_pred[pred_col] == 0)]
    ml_errors[lab] = {
        "n_falsos_negativos": len(fn),
        "ejemplos": fn[["Text", true_col, pred_col]].head(3).to_dict("records")
    }
report["falsos_negativos_etiquetas_criticas"] = ml_errors

# Vicarious: baja precisión -> revisar falsos positivos
vic_col = MULTILABEL_COLS[label_names.index("Vicarious")]
vic_fp = ml_pred[(ml_pred[vic_col] == 0) & (ml_pred["pred_Vicarious"] == 1)]
report["vicarious_falsos_positivos_n"] = len(vic_fp)
report["vicarious_falsos_positivos_ejemplos"] = vic_fp[["Text", vic_col, "pred_Vicarious"]].head(4).to_dict("records")

with open(f"{RES}/phase7_error_analysis.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print("FASE 7 COMPLETADA")
print(f"Errores críticos Severe->Mild: {report['n_errores_criticos_severe_a_mild']}")
print(f"Distribución errores de Severe: {report['distribucion_errores_de_severe']}")
print(f"Narrativas cortas (<=15 palabras) mal clasificadas: {report['n_narrativas_cortas_mal_clasificadas']} ({report['pct_narrativas_cortas_mal_clasificadas']*100:.1f}%)")
print(f"Confusión Medium<->High: {report['n_confusion_medium_high']} casos")
for lab in critical_labels:
    print(f"FN en {lab}: {ml_errors[lab]['n_falsos_negativos']}")
print(f"Vicarious FP: {report['vicarious_falsos_positivos_n']}")
