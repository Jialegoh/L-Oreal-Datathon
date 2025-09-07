import torch
from torch import nn
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

# -------------------------------
# 1. Define meta features (must match training)
# -------------------------------
meta_features = ["comment_length", "emoji_count", "mention_count", "lexical_richness", "readability"]

# -------------------------------
# 2. Model definition (copy from index2.py)
# -------------------------------
class CommentQualityModel(nn.Module):
    def __init__(self, transformer_name="xlm-roberta-base", meta_dim=5):
        super().__init__()
        from transformers import AutoModel
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hidden_size = self.transformer.config.hidden_size

        # Metadata branch
        self.meta_fc = nn.Linear(meta_dim, 32)

        # Combined layers
        self.dropout = nn.Dropout(0.2)
        self.fc_combined = nn.Linear(hidden_size + 32, 64)

        # Output layers
        self.out_quality = nn.Linear(64, 1)   # Regression
        self.out_relevance = nn.Linear(64, 1) # Regression
        self.out_spam = nn.Linear(64, 1)      # Binary classification (logit)

    def forward(self, input_ids, attention_mask, meta):
        # XLM-R pooled output
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output"):
            x = outputs.pooler_output
        else:
            x = outputs.last_hidden_state[:, 0]  # CLS token for XLM-R

        meta_x = torch.relu(self.meta_fc(meta))
        combined = torch.cat([x, meta_x], dim=1)
        combined = self.dropout(torch.relu(self.fc_combined(combined)))

        quality = self.out_quality(combined)
        relevance = self.out_relevance(combined)
        spam = self.out_spam(combined)

        return quality, relevance, spam

# -------------------------------
# 3. Load tokenizer + model weights
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

model = CommentQualityModel(
    transformer_name="xlm-roberta-base",
    meta_dim=len(meta_features)
).to(device)

# Load trained weights
model.load_state_dict(torch.load("best_comment_model.pt", map_location=device, weights_only=True))
model.eval()

# -------------------------------
# 4. Metadata extractor (same as training)
# -------------------------------
def extract_meta(text: str):
    text = str(text)
    return np.array([
        len(text),                               # comment_length
        sum(c in "😀😂😍😎🥰🤯👍💯🔥✨🙌🥳💖" for c in text),  # emoji_count
        text.count("@"),                         # mention_count
        len(set(text.split()))/max(1,len(text.split())), # lexical_richness
        len(text.split(".")),                    # simple readability proxy
    ], dtype=np.float32)

# -------------------------------
# 5. Prediction function
# -------------------------------
def predict_comment(text: str):
    meta = torch.tensor(extract_meta(text)).unsqueeze(0).to(device)

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        pred_quality, pred_relevance, pred_spam = model(input_ids, attention_mask, meta)

        # Convert outputs back to probabilities/values
        quality = torch.sigmoid(pred_quality).item()
        relevance = torch.sigmoid(pred_relevance).item()
        spam_prob = torch.sigmoid(pred_spam).item()

    return {
        "quality": quality,
        "relevance": relevance,
        "spam_probability": spam_prob
    }

# -------------------------------
# 6. Example usage
# -------------------------------
comments = [
    "Wow this video is amazing!",
    "CLICK HERE to win a free iPhone!!!",
    "Not really related to the topic"
]

results = [predict_comment(c) for c in comments]

df = pd.DataFrame({"comment": comments, "predictions": results})
print(df.to_string(index=False))
