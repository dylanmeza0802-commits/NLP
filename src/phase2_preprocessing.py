import sys, json
sys.path.insert(0, "/home/claude/proyecto/src")
from utils import *

set_seed()
RES = "/home/claude/proyecto/results"

mc_train, mc_devel, mc_test = load_multiclass()
ml_train, ml_devel, ml_test = load_multilabel()

# Limpieza para modelos tradicionales — se ajusta función determinística (no aprendida),
# por lo que puede aplicarse a los 3 splits sin fuga de información.
for df in (mc_train, mc_devel, mc_test):
    df["TEXT_CLEAN"] = df["TEXT"].apply(clean_text_traditional)
    df["TEXT_BETO"] = df["TEXT"].apply(clean_text_transformer)

for df in (ml_train, ml_devel, ml_test):
    df["TEXT_CLEAN"] = df["Text"].apply(clean_text_traditional)
    df["TEXT_BETO"] = df["Text"].apply(clean_text_transformer)

# Eliminación de duplicados EXACTOS solo dentro de train (nunca tocar devel/test)
before = len(mc_train)
mc_train_dedup = mc_train.drop_duplicates(subset=["TEXT_CLEAN"], keep="first")
after = len(mc_train_dedup)

info = {
    "duplicados_eliminados_train_multiclase": before - after,
    "filas_antes": before,
    "filas_despues": after,
    "nota": "Solo se deduplicó TRAIN. Devel y test se dejan intactos porque reflejan la distribución real de evaluación."
}

mc_train_dedup.to_csv(f"{RES}/mc_train_clean.csv", index=False)
mc_devel.to_csv(f"{RES}/mc_devel_clean.csv", index=False)
mc_test.to_csv(f"{RES}/mc_test_clean.csv", index=False)
ml_train.to_csv(f"{RES}/ml_train_clean.csv", index=False)
ml_devel.to_csv(f"{RES}/ml_devel_clean.csv", index=False)
ml_test.to_csv(f"{RES}/ml_test_clean.csv", index=False)

with open(f"{RES}/phase2_preprocessing_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print("FASE 2 COMPLETADA")
print(json.dumps(info, ensure_ascii=False, indent=2))
print("\nEjemplo de limpieza (traditional):")
print("ANTES:", mc_train_dedup['TEXT'].iloc[0][:150])
print("DESPUES:", mc_train_dedup['TEXT_CLEAN'].iloc[0][:150])
