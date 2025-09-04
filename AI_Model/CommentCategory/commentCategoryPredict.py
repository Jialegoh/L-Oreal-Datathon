import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Load model, tokenizer, and label binarizer classes
save_dir = "./saved_model"
model = BertForSequenceClassification.from_pretrained(save_dir)
tokenizer = BertTokenizer.from_pretrained(save_dir)
mlb_classes = np.load(f"{save_dir}/mlb_classes.npy", allow_pickle=True)

# Put model in evaluation mode
model.eval()

def predict(texts, threshold=0.5):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encodings).logits
        probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(int)
    results = []
    for row in preds:
        results.append([mlb_classes[i] for i, val in enumerate(row) if val == 1])
    return results

# Example usage
new_comments = [
    "This song is amazing, I love Asian pop music!",
    "Fashion trends are changing every season."
]

predicted_labels = predict(new_comments, threshold=0.3)  # try lower threshold for recall
for c, labels in zip(new_comments, predicted_labels):
    print(f"Comment: {c}\nPredicted Categories: {labels}\n")
