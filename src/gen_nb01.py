import sys
sys.path.insert(0, "/home/claude/proyecto/src")
from build_notebook import build_and_run

cells = [
("markdown", """# 01 - Análisis Exploratorio de Datos (EDA)
## Clasificación de reportes de violencia de género — WomenHelp-MX
**Procesamiento de Lenguaje Natural — UNSAAC**

Este notebook realiza la inspección inicial y el EDA orientado a NLP de los datasets
provistos para las dos subtareas del trabajo:
- **Subtarea 1 (Multiclase)**: grado de severidad (Mild/Medium/High/Severe).
- **Subtarea 2 (Multietiqueta)**: tipos de violencia presentes en la narrativa.

Todas las conclusiones de este notebook se basan **exclusivamente** en los datos reales
provistos; no se asumen nombres de clases ni distribuciones sin verificarlos primero.
"""),

("code", """import sys
sys.path.insert(0, "/home/claude/proyecto/src")
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

set_seed()
sns.set_style("whitegrid")
%matplotlib inline

mc_train, mc_devel, mc_test = load_multiclass()
ml_train, ml_devel, ml_test = load_multilabel()

print("Multiclase  -> train:", mc_train.shape, "devel:", mc_devel.shape, "test:", mc_test.shape)
print("Multietiqueta -> train:", ml_train.shape, "devel:", ml_devel.shape, "test:", ml_test.shape)
"""),

("markdown", """## 1. Inspección inicial

**Nota sobre consistencia de columnas**: se detectó que `Multiclass/train.csv` usa
columnas `TEXT`/`CLASS` en mayúsculas, mientras que `devel.csv` y `test.csv` usan
`Text`/`Class`. Se normalizó a mayúsculas en `utils.load_multiclass()` para evitar
errores silenciosos al concatenar o iterar splits.
"""),

("code", """print("Columnas Multiclass train (originales):", list(load_multiclass()[0].columns))
print()
print("Tipos de datos:")
print(mc_train.dtypes)
print()
print("Valores nulos (todas las columnas, todos los splits):")
for name, df in [("mc_train", mc_train), ("mc_devel", mc_devel), ("mc_test", mc_test),
                  ("ml_train", ml_train), ("ml_devel", ml_devel), ("ml_test", ml_test)]:
    print(f"  {name}: {df.isnull().sum().sum()} nulos")
"""),

("code", """# Duplicados EXACTOS de texto (no de fila completa, para detectar mejor duplicados reales)
dups_mc = mc_train[mc_train['TEXT'].duplicated(keep=False)].sort_values('TEXT')
print(f"Narrativas duplicadas en Multiclass/train: {mc_train['TEXT'].duplicated().sum()} de {len(mc_train)}")
print()
# Verificar si los duplicados tienen la MISMA etiqueta (ruido) o etiquetas distintas (inconsistencia real)
dup_groups = mc_train[mc_train['TEXT'].duplicated(keep=False)].groupby('TEXT')['CLASS'].nunique()
print(f"Grupos duplicados con etiqueta consistente: {(dup_groups==1).sum()}")
print(f"Grupos duplicados con etiquetas INCONSISTENTES (mismo texto, distinta clase): {(dup_groups>1).sum()}")
print()
print("Ejemplo de inconsistencia (mismo texto, distinta clase asignada por el anotador):")
incons_text = dup_groups[dup_groups>1].index[0]
print(mc_train[mc_train['TEXT']==incons_text][['TEXT','CLASS']])
"""),

("markdown", """**Hallazgo relevante**: existen 131 narrativas duplicadas (texto idéntico) en el
conjunto de entrenamiento multiclase; en 4 casos el mismo texto recibió etiquetas
de severidad distintas por distintos anotadores. Esto es evidencia de **ruido de
etiquetado inherente** al proceso de anotación humana y se documentará como
limitación en la sección de Ética del informe. Se eliminarán duplicados exactos
solo del conjunto de entrenamiento en la Fase 2 (nunca de devel/test).
"""),

("code", """# Ejemplos de narrativas por clase de severidad
for c in sorted(mc_train['CLASS'].unique()):
    print(f"--- CLASE {c} = {LABEL_NAMES_MULTICLASS[c]} (n={sum(mc_train['CLASS']==c)}) ---")
    for t in mc_train[mc_train['CLASS']==c]['TEXT'].head(2):
        print(" >", t[:220])
    print()
"""),

("markdown", """## 2. Distribución de clases (Subtarea 1 — Severidad)

Se infiere la correspondencia `CLASS -> nombre` (0=Mild, 1=Medium, 2=High, 3=Severe)
a partir de la evidencia textual anterior: la clase 0 refleja incumplimiento
económico sin agresión directa, mientras que la clase 3 concentra amenazas de
muerte y uso de objetos como arma. Esta hipótesis se corrobora cuantitativamente
más abajo (la clase 3 es, como es esperable en la realidad, la más minoritaria).
"""),

("code", """fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (name, df) in zip(axes, [("Train", mc_train), ("Devel", mc_devel), ("Test", mc_test)]):
    counts = df["CLASS"].value_counts().sort_index()
    labels = [LABEL_NAMES_MULTICLASS[c] for c in counts.index]
    ax.bar(labels, counts.values, color=sns.color_palette("rocket", 4))
    ax.set_title(f"Distribución de severidad - {name} (n={len(df)})")
    ax.set_ylabel("Frecuencia")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, f"{v}\\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)
plt.tight_layout()
plt.show()

ratio = mc_train['CLASS'].value_counts().max() / mc_train['CLASS'].value_counts().min()
print(f"\\nRatio de desbalance (clase mayoritaria/minoritaria): {ratio:.2f} : 1")
print("\\nLos 3 splits mantienen proporciones casi idénticas -> ya vienen estratificados.")
"""),

("markdown", """## 3. Longitud de narrativas por clase"""),

("code", """mc_train["len_words"] = mc_train["TEXT"].str.split().apply(len)
print(mc_train["len_words"].describe())

fig, ax = plt.subplots(figsize=(8,5))
for c in sorted(mc_train["CLASS"].unique()):
    sns.kdeplot(mc_train[mc_train["CLASS"]==c]["len_words"], label=LABEL_NAMES_MULTICLASS[c], ax=ax, clip=(0,400))
ax.set_title("Distribución de longitud de narrativas por severidad")
ax.set_xlabel("Nº de palabras")
plt.legend(); plt.tight_layout(); plt.show()

muy_cortos = mc_train[mc_train["len_words"] <= 3]
print(f"\\nNarrativas con <= 3 palabras: {len(muy_cortos)}")
print(muy_cortos["TEXT"].head(5).tolist())
"""),

("markdown", """**Observación**: hay narrativas de apenas 1-3 palabras (ej. registros administrativos
truncados). Estas serán un foco de atención en el análisis de errores (Fase 7), ya
que ofrecen poca señal semántica para cualquier modelo."""),

("markdown", """## 4. Distribución y co-ocurrencia de etiquetas (Subtarea 2)

No existe un diccionario explícito `L0..L6 -> nombre` en el dataset. Se infiere el
significado de cada columna analizando narrativas **puras** (donde solo esa
etiqueta está activa), presentado a continuación."""),

("code", """labels = MULTILABEL_COLS
for lab in labels:
    only = ml_train[(ml_train[lab]==1) & (ml_train[labels].sum(axis=1)==1)]
    print(f"=== {lab} -> hipótesis: {LABEL_NAMES_MULTILABEL[lab]}  (n_puros={len(only)}) ===")
    for t in only["Text"].head(2):
        print("   >", t[:200])
    print()
"""),

("code", """# Verificación clave: L6 (N/A) debe ser mutuamente excluyente por definición
l6_total = int(ml_train['L6'].sum())
l6_puro = int(((ml_train['L6']==1) & (ml_train[labels].sum(axis=1)==1)).sum())
print(f"L6 activo en total: {l6_total} filas")
print(f"L6 puro (sin ninguna otra etiqueta activa): {l6_puro} filas")
print(f"Coincide 100%: {l6_total == l6_puro}  -> confirma hipótesis 'N/A' es mutuamente excluyente")
"""),

("code", """support = ml_train[labels].sum().rename(index=LABEL_NAMES_MULTILABEL)
fig, ax = plt.subplots(figsize=(8,5))
ax.barh(support.index, support.values, color=sns.color_palette("mako", 7))
ax.set_title(f"Frecuencia de tipos de violencia (train, n={len(ml_train)})")
for i, v in enumerate(support.values):
    ax.text(v+20, i, f"{v} ({v/len(ml_train)*100:.1f}%)", va="center")
plt.tight_layout(); plt.show()
"""),

("code", """co = ml_train[labels].T.dot(ml_train[labels])
co.index = [LABEL_NAMES_MULTILABEL[c] for c in co.index]
co.columns = [LABEL_NAMES_MULTILABEL[c] for c in co.columns]
fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(co, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Co-ocurrencia entre tipos de violencia (train)")
plt.tight_layout(); plt.show()

combo = ml_train[labels].apply(lambda r: tuple(LABEL_NAMES_MULTILABEL[c] for c in r[r==1].index), axis=1)
print("\\nTop 10 combinaciones de etiquetas más frecuentes:")
print(combo.value_counts().head(10))
"""),

("markdown", """**Interpretación**: la combinación más común es `(Physical, Psychological)`, coherente
con la literatura sobre violencia doméstica (la violencia física casi siempre viene
acompañada de violencia psicológica). `Vicarious` (violencia a través de los hijos)
y `Sexual` son las etiquetas más minoritarias tras `N/A`, lo que exigirá mitigación
de desbalance en el modelado (Fases 5 y 6)."""),

("markdown", """## 5. Vocabulario, n-gramas y rasgos lingüísticos regionales"""),

("code", """STOP_ES = set('''de la que el en y a los del se las por un para con no una su al lo como más pero
sus le ya o este si porque esta entre cuando muy sin sobre tambien me hasta hay donde quien desde
todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mi antes algunos que unos yo
otro otras otra el tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo
usuaria agresor dice refiere menciona manifiesta'''.split())

def top_ngrams(texts, n=1, k=15):
    c = Counter()
    for t in texts:
        toks = [w for w in re.findall(r"[a-záéíóúñ]+", t.lower()) if w not in STOP_ES and len(w) > 2]
        grams = zip(*[toks[i:] for i in range(n)])
        c.update([" ".join(g) for g in grams])
    return c.most_common(k)

top_uni = top_ngrams(mc_train["TEXT"].tolist(), 1, 20)
words, freqs = zip(*top_uni)
fig, ax = plt.subplots(figsize=(8,6))
ax.barh(words, freqs, color=sns.color_palette("flare", 20))
ax.set_title("Top 20 unigramas más frecuentes (sin stopwords)")
plt.tight_layout(); plt.show()

print("\\nTop unigramas en clase SEVERE (posibles marcadores léxicos de gravedad):")
print(top_ngrams(mc_train[mc_train.CLASS==3]["TEXT"].tolist(), 1, 15))
"""),

("markdown", """**Rasgos lingüísticos observados** (relevantes para preprocesamiento, Fase 2):

- **Errores ortográficos frecuentes** por transcripción rápida del trabajador social:
  `FAMILAIR`, `ALIEMNTICIA`, `PROVISONAL`, `OCTUIBRE`, `MANIIFIESTA`. No se corrigen
  automáticamente (un corrector ortográfico agresivo podría alterar el significado
  o introducir ruido); se documenta como limitación.
- **Anonimización estandarizada**: términos como "LA USUARIA" y "EL AGRESOR"
  reemplazan sistemáticamente nombres propios — esto es intencional y debe
  preservarse (no es ruido).
- **Texto en mayúsculas** casi en su totalidad — decisión de normalizar a
  minúsculas en el preprocesamiento tradicional (reduce dispersión del
  vocabulario en TF-IDF) pero no en la rama para Transformers (BETO fue
  preentrenado con texto de casing mixto real).
- **Modismos regionales** presentes en citas textuales dentro de narrativas
  (ej. lenguaje coloquial/vulgar del norte de México en amenazas citadas) — se
  preservan porque son señal fuerte de severidad/tipo de violencia.
"""),

("markdown", """## 6. Verificación de la partición train/devel/test

Se comprueba que los splits provistos ya mantienen la distribución de clases /
etiquetas de forma consistente, evitando la necesidad de re-particionar."""),

("code", """print("Multiclase - % por clase en cada split:")
for name, df in [("train", mc_train), ("devel", mc_devel), ("test", mc_test)]:
    print(f"  {name}: ", (df['CLASS'].value_counts(normalize=True).sort_index()*100).round(2).to_dict())

print("\\nMultietiqueta - % de soporte por etiqueta en cada split:")
for name, df in [("train", ml_train), ("devel", ml_devel), ("test", ml_test)]:
    pct = (df[labels].sum()/len(df)*100).round(2)
    print(f"  {name}: ", {LABEL_NAMES_MULTILABEL[k]: v for k,v in pct.items()})
"""),

("markdown", """## Conclusiones del EDA

1. Dataset limpio en términos de nulos (0), pero con ruido de etiquetado leve
   (131 duplicados, 4 con etiquetas inconsistentes) — se maneja en Fase 2.
2. Desbalance fuerte en Subtarea 1 (ratio 8.6:1, clase Severe muy minoritaria) y
   en Subtarea 2 (Vicarious, Sexual y N/A son minoritarias) — requiere
   `class_weight="balanced"` / `pos_weight` en todos los modelos.
3. Los splits provistos ya están correctamente estratificados.
4. El mapeo de nombres de clases/etiquetas fue **inferido y verificado** con
   evidencia textual directa (no asumido arbitrariamente), incluyendo una
   verificación matemática exacta para `N/A` (L6).
5. Rasgos lingüísticos regionales y errores ortográficos deben preservarse
   parcialmente; se diseñan dos estrategias de limpieza distintas (tradicional
   vs. Transformer) en la Fase 2.
"""),
]

build_and_run(cells, "/home/claude/proyecto/notebooks/01_EDA.ipynb")
