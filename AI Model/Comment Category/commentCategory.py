import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# === Load Data ===
files = [
    "commentVideoMerged1.csv",
    "commentVideoMerged2.csv",
    "commentVideoMerged3.csv",
    "commentVideoMerged4.csv",
    "commentVideoMerged5.csv"
]

print("[Stage 1] Loading data...")
dfs = [pd.read_csv(f) for f in files]
data = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(data)} rows of data.")

# ⚡ Early subsampling for faster prototyping
sample_size = 50000   # adjust as needed
data = data.sample(sample_size, random_state=42).reset_index(drop=True)
print(f"Subsampled to {len(data)} rows for prototyping.")

# Parse labels
def parse_labels(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return [x]
    return []

print("[Stage 2] Parsing labels...")
data["labels"] = data["topicCategories_clean"].apply(parse_labels)

# Encode labels
print("[Stage 3] Encoding labels...")
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(data["labels"])
label_counts = Y.sum(axis=0)
valid_labels = [i for i, c in enumerate(label_counts) if c >= 100]
Y = Y[:, valid_labels]
mlb.classes_ = mlb.classes_[valid_labels]
print(f"Number of unique labels after filtering: {len(mlb.classes_)}")

# Clean text
print("[Stage 3.5] Cleaning text data...")
data["textOriginal"] = data["textOriginal"].fillna("").astype(str)

# Train/test split
print("[Stage 4] Splitting train/test data...")
X_train, X_test, y_train, y_test = train_test_split(
    data["textOriginal"], Y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# === BERT Pipeline ===
print("[Stage BERT-1] Preparing tokenizer and datasets...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ⚡ Subsample for faster prototyping
train_sample_size = 40000
test_sample_size = 10000

train_indices = np.random.choice(len(X_train), size=train_sample_size, replace=False)
test_indices = np.random.choice(len(X_test), size=test_sample_size, replace=False)

X_train_sample = X_train.iloc[train_indices]
y_train_sample = y_train[train_indices] if isinstance(y_train, np.ndarray) else y_train.iloc[train_indices]

X_test_sample = X_test.iloc[test_indices]
y_test_sample = y_test[test_indices] if isinstance(y_test, np.ndarray) else y_test.iloc[test_indices]

# Dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(X_train_sample, y_train_sample)
test_dataset = TextDataset(X_test_sample, y_test_sample)

# Model initialization
print("[Stage BERT-2] Initializing BERT model...")

# Model initialization

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification"
)

# === Compute class weights (inverse frequency, normalized) ===

from torch.nn import BCEWithLogitsLoss
class_counts = y_train_sample.sum(axis=0)
# Use pos_weight formula for BCEWithLogitsLoss
class_weights = (y_train_sample.shape[0] - class_counts) / (class_counts + 1e-6)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

# For focal loss, normalize alpha to [0,1]
alpha_focal = class_weights / class_weights.max()

# === Focal Loss (optional, for severe imbalance) ===
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce_loss)
        if self.alpha is not None:
            loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        else:
            loss = (1 - pt) ** self.gamma * bce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()

# === Custom Trainer with weighted BCE loss ===
from transformers import Trainer
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, use_focal=False, alpha_focal=None, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal = use_focal
        self.alpha_focal = alpha_focal
        self.sample_weights = sample_weights  # keep custom weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.use_focal:
            loss_fct = FocalLoss(alpha=self.alpha_focal)
        else:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if self.sample_weights is not None:
            sampler = WeightedRandomSampler(
                self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            return super().get_train_dataloader()

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    y_pred = (probs >= 0.5).astype(int)
    return {
        "micro/f1": f1_score(labels, y_pred, average="micro", zero_division=0),
        "macro/f1": f1_score(labels, y_pred, average="macro", zero_division=0),
        "micro/precision": precision_score(labels, y_pred, average="micro", zero_division=0),
        "micro/recall": recall_score(labels, y_pred, average="micro", zero_division=0),
    }

# Training arguments

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",   # correct key
    save_strategy="epoch",          # must match evaluation strategy
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="micro/f1",
    greater_is_better=True,
    report_to=[]
)

# Trainer



from torch.utils.data import DataLoader, WeightedRandomSampler
sample_weights = (y_train_sample @ class_weights_tensor.cpu().numpy()).astype(np.float32)

print("[Stage BERT-3] Starting training with class-weighted loss and batch sampler...")
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights_tensor,
    use_focal=False,  # set True to use focal loss
    alpha_focal=alpha_focal,
    data_collator=None,
    sample_weights=sample_weights  # ✅ pass here
)

trainer.train()

# Evaluation
print("[Stage BERT-4] Evaluating with default threshold 0.5...")
raw_preds = trainer.predict(test_dataset)
probs = torch.sigmoid(torch.tensor(raw_preds.predictions)).numpy()
y_pred_default = (probs >= 0.5).astype(int)
print(classification_report(y_test_sample, y_pred_default, target_names=mlb.classes_))

# Threshold tuning
print("[Stage BERT-5] Tuning thresholds per label...")
optimal_thresholds = []
y_pred_opt = np.zeros_like(probs)

for i in range(probs.shape[1]):
    best_thr, best_f1 = 0.5, 0
    for thr in np.linspace(0.1, 0.9, 9):
        preds = (probs[:, i] >= thr).astype(int)
        f1 = f1_score(y_test_sample[:, i], preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    optimal_thresholds.append(best_thr)
    y_pred_opt[:, i] = (probs[:, i] >= best_thr).astype(int)

print("Optimal thresholds per label:")

for label, thr in zip(mlb.classes_, optimal_thresholds):
    print(f" {label}: {thr:.2f}")


# Per-class F1 report (more detailed)
report = classification_report(y_test_sample, y_pred_opt, target_names=mlb.classes_, digits=3)
print(report)

# === Save model and tokenizer ===
save_dir = "./saved_model"

print(f"[Stage BERT-6] Saving model and tokenizer to {save_dir}...")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
np.save(f"{save_dir}/mlb_classes.npy", mlb.classes_)  # save label classes for decoding