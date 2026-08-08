import sys
sys.path.insert(0, "/home/claude/proyecto/src")
from build_notebook import build_and_run

cells = [
("markdown", """# 02 - Preprocesamiento
Diseñamos **dos estrategias distintas** de limpieza, justificadas académicamente:

| | Modelos tradicionales (TF-IDF) | Transformers (BETO) |
|---|---|---|
| Minúsculas | Sí (reduce dispersión de vocabulario) | No (BETO usa casing real del preentrenamiento) |
| Signos/caracteres especiales | Se filtran caracteres no alfanuméricos raros | Se preservan casi intactos |
| Stopwords | Se evalúan como hiperparámetro (no se eliminan por defecto) | No aplica (subword tokenization las maneja) |
| Lematización | No aplicada (el corpus es informal/con errores; lematizar sobre errores ortográficos introduce más ruido que beneficio) | No aplica |

**Prevención de data leakage**: cualquier transformación *aprendida* de los datos
(vectorizador TF-IDF, vocabulario) se ajusta ÚNICAMENTE sobre el conjunto de
entrenamiento, dentro de los Pipelines de scikit-learn usados en las Fases 3 y 5.
La limpieza de texto aquí es una función determinística (regex), no aprendida, por
lo que aplicarla igual a los 3 splits no constituye fuga de información.
"""),

("code", """import sys
sys.path.insert(0, "/home/claude/proyecto/src")
from utils import *
set_seed()

mc_train, mc_devel, mc_test = load_multiclass()
ml_train, ml_devel, ml_test = load_multilabel()

for df in (mc_train, mc_devel, mc_test):
    df["TEXT_CLEAN"] = df["TEXT"].apply(clean_text_traditional)
    df["TEXT_BETO"] = df["TEXT"].apply(clean_text_transformer)
for df in (ml_train, ml_devel, ml_test):
    df["TEXT_CLEAN"] = df["Text"].apply(clean_text_traditional)
    df["TEXT_BETO"] = df["Text"].apply(clean_text_transformer)

print("Ejemplo de limpieza:")
print("ORIGINAL       :", mc_train['TEXT'].iloc[0])
print("TRADICIONAL    :", mc_train['TEXT_CLEAN'].iloc[0])
print("BETO (mínima)  :", mc_train['TEXT_BETO'].iloc[0])
"""),

("code", """# Eliminación de duplicados EXACTOS -> SOLO en train (ver justificación en Fase 1 EDA)
before = len(mc_train)
mc_train = mc_train.drop_duplicates(subset=["TEXT_CLEAN"], keep="first")
after = len(mc_train)
print(f"Multiclase train: {before} -> {after} filas ({before-after} duplicados eliminados)")
print("Nota: devel y test se mantienen intactos porque representan la distribución real de evaluación.")
"""),

("code", """import os
os.makedirs("/home/claude/proyecto/results", exist_ok=True)
mc_train.to_csv("/home/claude/proyecto/results/mc_train_clean.csv", index=False)
mc_devel.to_csv("/home/claude/proyecto/results/mc_devel_clean.csv", index=False)
mc_test.to_csv("/home/claude/proyecto/results/mc_test_clean.csv", index=False)
ml_train.to_csv("/home/claude/proyecto/results/ml_train_clean.csv", index=False)
ml_devel.to_csv("/home/claude/proyecto/results/ml_devel_clean.csv", index=False)
ml_test.to_csv("/home/claude/proyecto/results/ml_test_clean.csv", index=False)
print("Datasets preprocesados guardados en results/*_clean.csv")
"""),

("markdown", """## Decisión: no se aplica corrección ortográfica automática

Se evaluó y descartó por dos razones:
1. Los correctores automáticos en español suelen "corregir" nombres/jerga regional
   hacia palabras del diccionario estándar, destruyendo información sociolingüística
   relevante para el análisis (uno de los objetivos explícitos del trabajo).
2. El volumen de errores (~decenas de patrones) es manejable por los propios
   modelos: TF-IDF con n-gramas de caracteres es robusto a variaciones ortográficas
   menores, y BETO (subword tokenization) también absorbe parte de esta variación.
"""),

("markdown", """## Decisión: stopwords y lematización como hiperparámetros, no como paso fijo

En violencia de género, ciertas "stopwords" convencionales del español
(negaciones como "no", pronombres posesivos) pueden ser señal relevante
(p. ej. "no le pega" vs "le pega"). Eliminarlas ciegamente arriesga perder
información semántica crítica. Por ello, en la Fase 3 se compara explícitamente
el desempeño con y sin n-gramas más amplios en lugar de imponer stopwords fijas.
"""),
]

build_and_run(cells, "/home/claude/proyecto/notebooks/02_Preprocesamiento.ipynb")
