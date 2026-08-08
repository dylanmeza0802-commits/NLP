"""
FASE 4 - Fine-tuning de BETO para clasificación multiclase (Subtarea 1)
=========================================================================
IMPORTANTE: Este script NO fue ejecutado por el asistente. Requiere GPU y
acceso a Hugging Face Hub, recursos no disponibles en el sandbox de este chat.
Ejecutar en Google Colab (Entorno de ejecución > Cambiar tipo de entorno > GPU).

Instrucciones de uso:
1. Sube a Colab: mc_train_clean.csv, mc_devel_clean.csv, mc_test_clean.csv
   (generados en la Fase 2, carpeta /home/claude/proyecto/results)
2. Ejecuta: !pip install transformers datasets accelerate -q
3. Corre este script completo.
4. Copia las métricas impresas al final hacia results/phase4_beto_multiclass_results.json
   para poder completar el informe y las tablas comparativas con datos reales.
"""
import numpy as np
import pandas as pd
import torch
import random
import json
from torch.utils.data import Dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, EarlyStoppingCallback)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

LABEL_NAMES = {0: "Mild", 1: "Medium", 2: "High", 3: "Severe"}

# --- Justificación del modelo preentrenado ---
# Se elige dccuchile/bert-base-spanish-wwm-uncased (BETO) porque:
#  1) Fue preentrenado desde cero en un corpus grande de español (no es una
#     traducción de BERT-EN), lo que captura mejor morfología y vocabulario
#     regional (relevante para modismos del norte de México).
#  2) Whole-Word-Masking (wwm) mejora la representación de palabras completas,
#     útil dado que el corpus tiene errores ortográficos que fragmentan
#     palabras en subtokens de forma distinta a la esperada.
#  3) Es un modelo "base" (110M parámetros): factible de entrenar con recursos
#     limitados (una sola GPU T4 de Colab) en tiempos razonables (<1h por tarea).
# Alternativa si BETO no está disponible o los recursos son aún más limitados:
#  - "PlanTL-GOB-ES/roberta-base-bne" (RoBERTa entrenado en español).
#  - "bert-base-multilingual-cased" (mBERT) como alternativa multilingüe.
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
MAX_LENGTH = 160   # cubre el percentil ~90 de longitud de narrativas (ver EDA Fase 1)
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 6         # con early stopping, rara vez se completan todas
WEIGHT_DECAY = 0.01

RES = "/home/claude/proyecto/results"  # ajustar ruta en Colab si es necesario

train = pd.read_csv(f"{RES}/mc_train_clean.csv")
devel = pd.read_csv(f"{RES}/mc_devel_clean.csv")
test = pd.read_csv(f"{RES}/mc_test_clean.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ViolenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(list(texts), truncation=True, padding="max_length",
                                    max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_ds = ViolenceDataset(train["TEXT_BETO"], train["CLASS"], tokenizer, MAX_LENGTH)
devel_ds = ViolenceDataset(devel["TEXT_BETO"], devel["CLASS"], tokenizer, MAX_LENGTH)
test_ds = ViolenceDataset(test["TEXT_BETO"], test["CLASS"], tokenizer, MAX_LENGTH)

# --- Pesos de clase (por el desbalance detectado en la Fase 1: ratio ~8.6:1) ---
class_counts = train["CLASS"].value_counts().sort_index().values
class_weights = torch.tensor((class_counts.sum() / (len(class_counts) * class_counts)), dtype=torch.float)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_micro": f1_score(labels, preds, average="micro"),
    }

args = TrainingArguments(
    output_dir="./beto_multiclass_ckpt",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=0.1,               # scheduler: warmup lineal + decay lineal (default de Trainer)
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=SEED,
    logging_steps=50,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=devel_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# ---- Evaluación final en TEST ----
test_out = trainer.predict(test_ds)
test_preds = np.argmax(test_out.predictions, axis=1)
test_labels = test["CLASS"].values

results = {
    "model": MODEL_NAME,
    "hparams": {"max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
                "epochs_max": EPOCHS, "weight_decay": WEIGHT_DECAY},
    "test_metrics": {
        "accuracy": accuracy_score(test_labels, test_preds),
        "f1_macro": f1_score(test_labels, test_preds, average="macro"),
        "f1_micro": f1_score(test_labels, test_preds, average="micro"),
        "precision_macro": precision_score(test_labels, test_preds, average="macro"),
        "recall_macro": recall_score(test_labels, test_preds, average="macro"),
    },
    "classification_report": classification_report(test_labels, test_preds,
        target_names=[LABEL_NAMES[i] for i in range(4)], output_dict=True),
    "confusion_matrix": confusion_matrix(test_labels, test_preds).tolist(),
}

with open(f"{RES}/phase4_beto_multiclass_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print(json.dumps(results["test_metrics"], indent=2))

# Guarda también las predicciones para el análisis de errores (Fase 7)
test_pred_df = test.copy()
test_pred_df["pred_beto"] = test_preds
test_pred_df.to_csv(f"{RES}/mc_test_predictions_beto.csv", index=False)
