import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, accuracy_score

print("[Stage 0] Settings")
# -------------------------------
# 0. Settings
# -------------------------------
MODEL_NAME = "xlm-roberta-base"   # keep this or swap to a smaller multilingual model for faster iterations
BATCH_SIZE = 8
NUM_EPOCHS = 5
LR = 2e-5
VAL_SIZE = 0.2
PATIENCE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[Stage 1] Load dataset + simple clean")
# -------------------------------
# 1. Load dataset + simple clean
# -------------------------------
df = pd.read_csv("comment_quality_results.csv")
df["quality_score"] = df["CQS"].astype(float).fillna(0.0)
df["relevance_score"] = df["relevance"].astype(float).fillna(0.0)

# ensure spam_label exists (your existing function)
def aggressive_pseudo_spam_label(text):
    text = str(text).lower()
    if "http" in text or "www" in text or ".com" in text:
        return 1
    if len(text) < 8 or sum(c.isalpha() for c in text)/max(len(text),1) < 0.5:
        return 1
    if text.count("!") > 3 or text.count("?") > 3 or any(c*3 in text for c in text):
        return 1
    emojis = sum(c in "😀😂😍😎🥰🤯👍💯🔥✨🙌🥳💖" for c in text)
    if emojis > 5:
        return 1
    spam_keywords = ["buy now", "free", "click here", "subscribe", "winner", "offer"]
    if any(k in text for k in spam_keywords):
        return 1
    return 0

df["spam_label"] = df["textOriginal"].apply(aggressive_pseudo_spam_label).astype(float)

# Optional: fill/compute meta features if any are NaN
meta_features = ["comment_length", "emoji_count", "mention_count", "lexical_richness", "readability"]
for f in meta_features:
    if f not in df.columns:
        df[f] = 0.0
df[meta_features] = df[meta_features].fillna(0.0).astype(np.float32)

print("[Stage 2] Train/validation split")
# -------------------------------
# 2. Train/validation split
# -------------------------------
train_df, val_df = train_test_split(df, test_size=VAL_SIZE, random_state=42, shuffle=True)
train_df = train_df.sample(5000, random_state=42)  # only 5k samples for dev
val_df = val_df.sample(1000, random_state=42)

print("[Stage 3] Scale regression targets (fit on train only)")
# -------------------------------
# 3. Scale regression targets (fit on train only)
# -------------------------------
scaler = StandardScaler()
train_targets = train_df[["quality_score", "relevance_score"]].values
scaler.fit(train_targets)
train_df[["quality_score", "relevance_score"]] = scaler.transform(train_targets)
val_df[["quality_score", "relevance_score"]] = scaler.transform(val_df[["quality_score", "relevance_score"]].values)

print("[Stage 4] Dataset class (unchanged) but will use train/val dfs")
# -------------------------------
# 4. Dataset class (unchanged) but will use train/val dfs
# -------------------------------
class CommentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["textOriginal"].astype(str).tolist()
        self.meta = df[meta_features].values.astype(np.float32)
        self.quality = df["quality_score"].values.astype(np.float32)
        self.relevance = df["relevance_score"].values.astype(np.float32)
        self.spam = df["spam_label"].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        meta = torch.tensor(self.meta[idx])
        quality = torch.tensor(self.quality[idx])
        relevance = torch.tensor(self.relevance[idx])
        spam = torch.tensor(self.spam[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "meta": meta,
            "quality": quality,
            "relevance": relevance,
            "spam": spam
        }

print("[Stage 5] Tokenizer / Dataloaders")
# -------------------------------
# 5. Tokenizer / Dataloaders
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = CommentDataset(train_df, tokenizer)
val_dataset = CommentDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("[Stage 6] Model (no sigmoids for reg; spam returns logits)")
# -------------------------------
# 6. Model (no sigmoids for reg; spam returns logits)
# -------------------------------
class CommentQualityModel(nn.Module):
    def __init__(self, transformer_name=MODEL_NAME, meta_dim=len(meta_features)):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hidden_size = self.transformer.config.hidden_size

        self.meta_fc = nn.Linear(meta_dim, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc_combined = nn.Linear(hidden_size + 32, 64)

        # outputs: raw (regression) and raw spam logits
        self.out_quality = nn.Linear(64, 1)
        self.out_relevance = nn.Linear(64, 1)
        self.out_spam = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask, meta):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[1]   # pooled output (works for models with pooler)
        meta_x = torch.relu(self.meta_fc(meta))
        combined = torch.cat([pooled, meta_x], dim=1)
        combined = self.dropout(torch.relu(self.fc_combined(combined)))

        quality = self.out_quality(combined)     # raw (to use with MSE on scaled targets)
        relevance = self.out_relevance(combined) # raw
        spam_logits = self.out_spam(combined)    # raw logits (BCEWithLogitsLoss)

        return quality, relevance, spam_logits

model = CommentQualityModel().to(DEVICE)

print("[Stage 7] Losses, optimizer, pos_weight for spam")
# -------------------------------
# 7. Losses, optimizer, pos_weight for spam
# -------------------------------
# compute pos_weight from training labels to help class imbalance
pos = train_df["spam_label"].sum()
neg = len(train_df) - pos
pos_weight = torch.tensor([(neg / (pos + 1e-8))]).to(DEVICE)

loss_fn_reg = nn.MSELoss()
loss_fn_class = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = AdamW(model.parameters(), lr=LR)

# weights for multi-task loss (tune these)
w_q, w_r, w_s = 0.5, 0.5, 1.0

# AMP
scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

print("[Stage 8] Training + Validation loop with early stopping")
# -------------------------------
# 8. Training + Validation loop with early stopping
# -------------------------------
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_losses = []
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        meta = batch["meta"].to(DEVICE)
        quality = batch["quality"].to(DEVICE).unsqueeze(1)
        relevance = batch["relevance"].to(DEVICE).unsqueeze(1)
        spam = batch["spam"].to(DEVICE).unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pq, pr, ps_logits = model(input_ids, attention_mask, meta)
            loss_q = loss_fn_reg(pq, quality)
            loss_r = loss_fn_reg(pr, relevance)
            loss_s = loss_fn_class(ps_logits, spam)
            loss = w_q*loss_q + w_r*loss_r + w_s*loss_s

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        train_losses.append(loss.item())
        pbar.set_postfix(train_loss=np.mean(train_losses))

    # ---------------- Validation
    model.eval()
    val_losses = []
    all_q_pred, all_r_pred, all_s_pred = [], [], []
    all_q_true, all_r_true, all_s_true = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            meta = batch["meta"].to(DEVICE)
            quality = batch["quality"].to(DEVICE).unsqueeze(1)
            relevance = batch["relevance"].to(DEVICE).unsqueeze(1)
            spam = batch["spam"].to(DEVICE).unsqueeze(1)

            pq, pr, ps_logits = model(input_ids, attention_mask, meta)
            loss_q = loss_fn_reg(pq, quality)
            loss_r = loss_fn_reg(pr, relevance)
            loss_s = loss_fn_class(ps_logits, spam)
            loss = w_q*loss_q + w_r*loss_r + w_s*loss_s
            val_losses.append(loss.item())

            # collect preds (move to cpu & numpy)
            all_q_pred.append(pq.squeeze(-1).cpu().numpy())
            all_r_pred.append(pr.squeeze(-1).cpu().numpy())
            all_s_pred.append(torch.sigmoid(ps_logits.squeeze(-1)).cpu().numpy())

            all_q_true.append(quality.squeeze(-1).cpu().numpy())
            all_r_true.append(relevance.squeeze(-1).cpu().numpy())
            all_s_true.append(spam.squeeze(-1).cpu().numpy())

    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1} — train_loss: {np.mean(train_losses):.4f}  val_loss: {val_loss:.4f}")

    # concat arrays
    all_q_pred = np.concatenate(all_q_pred, axis=0)
    all_r_pred = np.concatenate(all_r_pred, axis=0)
    all_s_pred = np.concatenate(all_s_pred, axis=0)

    all_q_true = np.concatenate(all_q_true, axis=0)
    all_r_true = np.concatenate(all_r_true, axis=0)
    all_s_true = np.concatenate(all_s_true, axis=0)

    # inverse transform regression preds/targets to original scale for metrics
    val_preds_comb = np.vstack([all_q_pred, all_r_pred]).T
    val_true_comb = np.vstack([all_q_true, all_r_true]).T
    inv_preds = scaler.inverse_transform(val_preds_comb)
    inv_true = scaler.inverse_transform(val_true_comb)

    q_pred_inv = inv_preds[:, 0]; r_pred_inv = inv_preds[:, 1]
    q_true_inv = inv_true[:, 0]; r_true_inv = inv_true[:, 1]

    # regression metrics (on original scale)
    q_mae = mean_absolute_error(q_true_inv, q_pred_inv)
    q_rmse = mean_squared_error(q_true_inv, q_pred_inv, squared=False)
    r_mae = mean_absolute_error(r_true_inv, r_pred_inv)
    r_rmse = mean_squared_error(r_true_inv, r_pred_inv, squared=False)

    # classification metrics (spam) — choose threshold 0.5
    s_pred_bin = (all_s_pred >= 0.5).astype(int)
    s_true = all_s_true.astype(int)

    s_acc = accuracy_score(s_true, s_pred_bin)
    s_prec = precision_score(s_true, s_pred_bin, zero_division=0)
    s_rec = recall_score(s_true, s_pred_bin, zero_division=0)
    s_f1 = f1_score(s_true, s_pred_bin, zero_division=0)

    print(f"Quality MAE/RMSE: {q_mae:.4f}/{q_rmse:.4f} | Relevance MAE/RMSE: {r_mae:.4f}/{r_rmse:.4f}")
    print(f"Spam acc/prec/rec/f1: {s_acc:.4f}/{s_prec:.4f}/{s_rec:.4f}/{s_f1:.4f}")

    # early stopping & checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_comment_model.pt")
        patience_counter = 0
        print("Saved best model.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print("[Stage 9] After training: quick inference example on val (re-use code above)")
# -------------------------------
# 9. After training: quick inference example on val (re-use code above)
# -------------------------------
# load model if you need to
# model.load_state_dict(torch.load("best_comment_model.pt"))
# model.eval()

print("Training finished. Best val loss:", best_val_loss)
