"""
FASE 6 - Fine-tuning de BETO para clasificación multietiqueta (Subtarea 2)
=========================================================================
IMPORTANTE: No ejecutado por el asistente (sin GPU / sin acceso a Hugging Face
Hub en este sandbox). Ejecutar en Google Colab con GPU.

Diferencia clave vs. Fase 4 (explicar en la exposición):
- Multiclase: una sola etiqueta correcta por ejemplo -> Softmax + CrossEntropyLoss
  (las probabilidades de las 4 clases suman 1, son mutuamente excluyentes).
- Multietiqueta: 0 a N etiquetas correctas por ejemplo, no excluyentes entre sí
  -> Sigmoid independiente por etiqueta + BCEWithLogitsLoss (cada etiqueta es
  un problema de clasificación binaria independiente comparten el encoder).
  Por eso NUNCA se usa argmax aquí: se usa sigmoid + umbral por etiqueta.
"""
import numpy as np
import pandas as pd
import torch
import random
import json
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, EarlyStoppingCallback)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MULTILABEL_COLS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]
LABEL_NAMES = {"L0": "Economic", "L1": "Physical", "L2": "Property-related",
               "L3": "Psychological", "L4": "Sexual", "L5": "Vicarious", "L6": "N/A"}
label_list = [LABEL_NAMES[c] for c in MULTILABEL_COLS]
NUM_LABELS = len(MULTILABEL_COLS)

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
MAX_LENGTH = 160
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 8
WEIGHT_DECAY = 0.01

RES = "/home/claude/proyecto/results"

train = pd.read_csv(f"{RES}/ml_train_clean.csv")
devel = pd.read_csv(f"{RES}/ml_devel_clean.csv")
test = pd.read_csv(f"{RES}/ml_test_clean.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class MultilabelDataset(Dataset):
    def __init__(self, texts, labels_matrix, tokenizer, max_length):
        self.encodings = tokenizer(list(texts), truncation=True, padding="max_length",
                                    max_length=max_length)
        self.labels = labels_matrix.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_ds = MultilabelDataset(train["TEXT_BETO"], train[MULTILABEL_COLS].values, tokenizer, MAX_LENGTH)
devel_ds = MultilabelDataset(devel["TEXT_BETO"], devel[MULTILABEL_COLS].values, tokenizer, MAX_LENGTH)
test_ds = MultilabelDataset(test["TEXT_BETO"], test[MULTILABEL_COLS].values, tokenizer, MAX_LENGTH)

# Pesos por etiqueta (pos_weight de BCEWithLogitsLoss) según desbalance detectado en Fase 1
support = train[MULTILABEL_COLS].sum().values
n = len(train)
pos_weight = torch.tensor((n - support) / np.clip(support, 1, None), dtype=torch.float)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, problem_type="multi_label_classification"
)

class WeightedMultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int)  # umbral provisional durante entrenamiento;
                                          # el umbral óptimo por etiqueta se calcula DESPUÉS, solo con devel.
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(labels, preds),
    }

args = TrainingArguments(
    output_dir="./beto_multilabel_ckpt",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=SEED,
    logging_steps=50,
    report_to="none",
)

trainer = WeightedMultilabelTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=devel_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# ---- Umbral óptimo por etiqueta, calculado SOLO con devel ----
devel_out = trainer.predict(devel_ds)
devel_probs = sigmoid(devel_out.predictions)
devel_labels = devel[MULTILABEL_COLS].values

thresholds = {}
for i, lab in enumerate(label_list):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(devel_labels[:, i], (devel_probs[:, i] >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    thresholds[lab] = float(best_t)

# ---- Evaluación final en TEST con umbrales óptimos ----
test_out = trainer.predict(test_ds)
test_probs = sigmoid(test_out.predictions)
test_labels = test[MULTILABEL_COLS].values
thr_arr = np.array([thresholds[l] for l in label_list])
test_preds = (test_probs >= thr_arr).astype(int)

results = {
    "model": MODEL_NAME,
    "hparams": {"max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
                "epochs_max": EPOCHS, "weight_decay": WEIGHT_DECAY},
    "thresholds_optimos_devel": thresholds,
    "test_metrics": {
        "f1_macro": f1_score(test_labels, test_preds, average="macro", zero_division=0),
        "f1_micro": f1_score(test_labels, test_preds, average="micro", zero_division=0),
        "precision_macro": precision_score(test_labels, test_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(test_labels, test_preds, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(test_labels, test_preds),
    },
}

per_label = {}
for i, lab in enumerate(label_list):
    per_label[lab] = {
        "precision": float(precision_score(test_labels[:, i], test_preds[:, i], zero_division=0)),
        "recall": float(recall_score(test_labels[:, i], test_preds[:, i], zero_division=0)),
        "f1": float(f1_score(test_labels[:, i], test_preds[:, i], zero_division=0)),
    }
results["metricas_por_etiqueta"] = per_label

with open(f"{RES}/phase6_beto_multilabel_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print(json.dumps(results["test_metrics"], indent=2))
