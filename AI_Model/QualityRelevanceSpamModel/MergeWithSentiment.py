import torch
from torch import nn
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm

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
# 6. Process comments_with_sentiment.csv
# -------------------------------
def process_csv_comments(input_file="comments_with_sentiment.csv", output_file="comments_evaluated.csv", test_mode=True):
    print(f"Loading data from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} comments")
    print(f"Columns available: {list(df.columns)}")
    
    # Check if textOriginal column exists
    if 'textOriginal' not in df.columns:
        print("Error: 'textOriginal' column not found in the CSV file")
        return
    
    # TEMPORARY: Process only first batch for testing
    if test_mode:
        print("TEST MODE: Processing only first 1000 comments")
        df = df.head(1000).copy()
        output_file = "comments_evaluated_test.csv"
    
    # Process comments in batches to avoid memory issues
    batch_size = 1000
    all_results = []
    
    # Calculate total batches for progress bar
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches", total=total_batches):
        batch_end = min(i + batch_size, len(df))
        
        batch_df = df.iloc[i:batch_end].copy()
        batch_results = []
        
        # Process individual comments with progress bar
        for idx, row in tqdm(batch_df.iterrows(), desc=f"Batch {i//batch_size + 1}", total=len(batch_df), leave=False):
            text = str(row['textOriginal']) if pd.notna(row['textOriginal']) else ""
            
            # Skip empty comments
            if not text.strip():
                result = {
                    "quality": 0.0,
                    "relevance": 0.0,
                    "spam_probability": 0.5
                }
            else:
                try:
                    result = predict_comment(text)
                except Exception as e:
                    print(f"Error processing comment at index {idx}: {e}")
                    result = {
                        "quality": 0.0,
                        "relevance": 0.0,
                        "spam_probability": 0.5
                    }
            
            batch_results.append(result)
        
        all_results.extend(batch_results)
    
    # Add results to the dataframe
    df['quality_score'] = [r['quality'] for r in all_results]
    df['relevance_score'] = [r['relevance'] for r in all_results]
    df['spam_probability'] = [r['spam_probability'] for r in all_results]
    
    # Add categorical classifications
    df['quality_category'] = df['quality_score'].apply(lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low')
    df['relevance_category'] = df['relevance_score'].apply(lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low')
    df['is_spam'] = df['spam_probability'].apply(lambda x: 'Yes' if x > 0.5 else 'No')
    
    # Save results
    print(f"Saving results to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"\nEvaluation complete! Results saved to {output_file}")
    print(f"Summary statistics:")
    print(f"Quality - High: {sum(df['quality_category']=='High')}, Medium: {sum(df['quality_category']=='Medium')}, Low: {sum(df['quality_category']=='Low')}")
    print(f"Relevance - High: {sum(df['relevance_category']=='High')}, Medium: {sum(df['relevance_category']=='Medium')}, Low: {sum(df['relevance_category']=='Low')}")
    print(f"Spam - Yes: {sum(df['is_spam']=='Yes')}, No: {sum(df['is_spam']=='No')}")
    print(f"Average quality score: {df['quality_score'].mean():.3f}")
    print(f"Average relevance score: {df['relevance_score'].mean():.3f}")
    print(f"Average spam probability: {df['spam_probability'].mean():.3f}")

if __name__ == "__main__":
    # Set test_mode=True to process only first 1000 comments
    # Set test_mode=False to process all comments
    process_csv_comments(test_mode=False)
