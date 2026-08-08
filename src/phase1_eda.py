import sys, json, re
sys.path.insert(0, "/home/claude/proyecto/src")
from utils import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

set_seed()
sns.set_style("whitegrid")
FIG = "/home/claude/proyecto/figures"
RES = "/home/claude/proyecto/results"

mc_train, mc_devel, mc_test = load_multiclass()
ml_train, ml_devel, ml_test = load_multilabel()

summary = {}

# ---- 1. Dimensiones y estructura ----
summary["dimensiones"] = {
    "multiclass_train": mc_train.shape, "multiclass_devel": mc_devel.shape, "multiclass_test": mc_test.shape,
    "multilabel_train": ml_train.shape, "multilabel_devel": ml_devel.shape, "multilabel_test": ml_test.shape,
}

# ---- 2. Nulos y duplicados ----
summary["nulos_duplicados"] = {
    "multiclass_train_nulls": int(mc_train.isnull().sum().sum()),
    "multiclass_train_dups": int(mc_train.duplicated(subset=["TEXT"]).sum()),
    "multilabel_train_nulls": int(ml_train.isnull().sum().sum()),
    "multilabel_train_dups": int(ml_train.duplicated(subset=["Text"]).sum()),
}

# ---- 3. Distribución de clases (Subtarea 1) ----
for name, df in [("train", mc_train), ("devel", mc_devel), ("test", mc_test)]:
    df["len_words"] = df["TEXT"].str.split().apply(len)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (name, df) in zip(axes, [("Train", mc_train), ("Devel", mc_devel), ("Test", mc_test)]):
    counts = df["CLASS"].value_counts().sort_index()
    labels = [LABEL_NAMES_MULTICLASS[c] for c in counts.index]
    sns.barplot(x=labels, y=counts.values, ax=ax, palette="rocket")
    ax.set_title(f"Distribución de severidad - {name} (n={len(df)})")
    ax.set_ylabel("Frecuencia")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG}/01_distribucion_severidad.png", dpi=130)
plt.close()

summary["distribucion_clases_train"] = mc_train["CLASS"].value_counts().sort_index().to_dict()
summary["distribucion_clases_devel"] = mc_devel["CLASS"].value_counts().sort_index().to_dict()
summary["distribucion_clases_test"] = mc_test["CLASS"].value_counts().sort_index().to_dict()

# Índice de desbalance
counts = mc_train["CLASS"].value_counts()
summary["ratio_desbalance_multiclase"] = float(counts.max() / counts.min())

# ---- 4. Longitud de narrativas ----
fig, ax = plt.subplots(figsize=(8, 5))
for c in sorted(mc_train["CLASS"].unique()):
    sns.kdeplot(mc_train[mc_train["CLASS"] == c]["len_words"], label=LABEL_NAMES_MULTICLASS[c], ax=ax, clip=(0, 400))
ax.set_title("Distribución de longitud de narrativas por clase de severidad")
ax.set_xlabel("Nº de palabras")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/02_longitud_por_clase.png", dpi=130)
plt.close()

summary["longitud_texto_train"] = mc_train["len_words"].describe().to_dict()
summary["longitud_por_clase"] = mc_train.groupby("CLASS")["len_words"].describe()[["mean","50%","min","max"]].to_dict()

# textos extremadamente cortos
muy_cortos = mc_train[mc_train["len_words"] <= 3]
summary["n_textos_muy_cortos_le3_palabras"] = len(muy_cortos)
summary["ejemplos_textos_cortos"] = muy_cortos["TEXT"].head(5).tolist()

# ---- 5. Distribución multietiqueta ----
support = ml_train[MULTILABEL_COLS].sum().rename(index=LABEL_NAMES_MULTILABEL)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=support.values, y=support.index, ax=ax, palette="mako", orient="h")
ax.set_title(f"Frecuencia de tipos de violencia (Multilabel train, n={len(ml_train)})")
ax.set_xlabel("Nº de narrativas con la etiqueta activa")
for i, v in enumerate(support.values):
    ax.text(v + 20, i, f"{v} ({v/len(ml_train)*100:.1f}%)", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG}/03_distribucion_etiquetas.png", dpi=130)
plt.close()

summary["soporte_etiquetas_train"] = support.to_dict()

# Co-ocurrencia
co = ml_train[MULTILABEL_COLS].T.dot(ml_train[MULTILABEL_COLS])
co.index = [LABEL_NAMES_MULTILABEL[c] for c in co.index]
co.columns = [LABEL_NAMES_MULTILABEL[c] for c in co.columns]
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(co, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Matriz de co-ocurrencia entre tipos de violencia (train)")
plt.tight_layout()
plt.savefig(f"{FIG}/04_coocurrencia_etiquetas.png", dpi=130)
plt.close()

# combinaciones más frecuentes
combo = ml_train[MULTILABEL_COLS].apply(lambda r: tuple(LABEL_NAMES_MULTILABEL[c] for c in r[r == 1].index), axis=1)
top_combos = combo.value_counts().head(10)
summary["combinaciones_mas_frecuentes"] = {str(k): int(v) for k, v in top_combos.items()}

# nº etiquetas por narrativa
nlab = ml_train[MULTILABEL_COLS].sum(axis=1)
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=nlab, ax=ax, palette="crest")
ax.set_title("Número de tipos de violencia por narrativa (train)")
ax.set_xlabel("Nº de etiquetas activas")
plt.tight_layout()
plt.savefig(f"{FIG}/05_num_etiquetas_por_fila.png", dpi=130)
plt.close()

# Verificación de que L6 (N/A) es mutuamente excluyente -> confirma hipótesis de mapeo
l6_puro = int(((ml_train["L6"] == 1) & (ml_train[MULTILABEL_COLS].sum(axis=1) == 1)).sum())
summary["verificacion_L6_NA_mutuamente_excluyente"] = {
    "total_L6_activo": int(ml_train["L6"].sum()),
    "L6_puro_sin_otras_etiquetas": l6_puro,
    "coincide": l6_puro == int(ml_train["L6"].sum())
}

# ---- 6. Verificación de que los splits provistos ya están estratificados ----
def dist_pct(df, col="CLASS"):
    return (df[col].value_counts(normalize=True).sort_index() * 100).round(2)

summary["verificacion_estratificacion_multiclase"] = {
    "train_%": dist_pct(mc_train).to_dict(),
    "devel_%": dist_pct(mc_devel).to_dict(),
    "test_%": dist_pct(mc_test).to_dict(),
}

ml_support_pct = {}
for name, df in [("train", ml_train), ("devel", ml_devel), ("test", ml_test)]:
    pct = (df[MULTILABEL_COLS].sum() / len(df) * 100).round(2)
    ml_support_pct[name] = {LABEL_NAMES_MULTILABEL[k]: v for k, v in pct.items()}
summary["verificacion_estratificacion_multilabel_%"] = ml_support_pct

# ---- 7. N-gramas / vocabulario frecuente (rápido, con stopwords español básicas) ----
STOP_ES = set("""de la que el en y a los del se las por un para con no una su al lo como más pero
sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde
todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo
otro otras otra él tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo
nosotros mi mis tú te ti tu tus ellas nosotras vosotros vosotras os mío mía míos mías tuyo tuya tuyos
tuyas suyo suya suyos suyas nuestro nuestra nuestros nuestras vuestro vuestra vuestros vuestras esos
esas usuaria agresor dice refiere menciona manifiesta""".split())

def top_ngrams(texts, n=1, k=20):
    c = Counter()
    for t in texts:
        toks = [w for w in re.findall(r"[a-záéíóúñ]+", t.lower()) if w not in STOP_ES and len(w) > 2]
        grams = zip(*[toks[i:] for i in range(n)])
        c.update([" ".join(g) for g in grams])
    return c.most_common(k)

summary["top_unigramas_generales"] = top_ngrams(mc_train["TEXT"].tolist(), 1, 20)
summary["top_bigramas_generales"] = top_ngrams(mc_train["TEXT"].tolist(), 2, 15)
summary["top_unigramas_severe"] = top_ngrams(mc_train[mc_train.CLASS == 3]["TEXT"].tolist(), 1, 15)
summary["top_unigramas_mild"] = top_ngrams(mc_train[mc_train.CLASS == 0]["TEXT"].tolist(), 1, 15)

# wordcloud-like bar for general unigrams
words, freqs = zip(*summary["top_unigramas_generales"])
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=list(freqs), y=list(words), palette="flare", ax=ax)
ax.set_title("Top 20 unigramas más frecuentes (train, sin stopwords)")
plt.tight_layout()
plt.savefig(f"{FIG}/06_top_unigramas.png", dpi=130)
plt.close()

# ---- 8. Errores ortográficos / informalidad (indicador aproximado) ----
# Heurística: palabras con repeticiones de letra anómalas o patrones típicos de error de tecleo rápido
sample_errors = ["FAMILAIR","ALIEMNTICIA","PROVISONAL","OCTUIBRE","MANIIFIESTA","DAPENSION","VALLAY"]
summary["ejemplos_errores_ortograficos_detectados_manualmente"] = sample_errors

with open(f"{RES}/phase1_eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print("FASE 1 COMPLETADA")
print(json.dumps({k: v for k, v in summary.items() if k in [
    "dimensiones","nulos_duplicados","distribucion_clases_train","ratio_desbalance_multiclase",
    "verificacion_L6_NA_mutuamente_excluyente"]}, ensure_ascii=False, indent=2))
