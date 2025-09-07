import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

# === 1. Load data ===
files = [f"commentVideoMerged{i}.csv" for i in range(1, 6)]
dfs = [pd.read_csv(f) for f in files]
data = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(data)} comments.")

# === 2. Load multilingual sentiment model ===
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

# === 3. Run sentiment analysis with tqdm progress bar ===
texts = data["textOriginal"].fillna("").astype(str).tolist()
batch_size = 8
results = []
print("Starting sentiment analysis loop...")
for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
	batch_texts = texts[i:i+batch_size]
	batch_results = sentiment_pipeline(batch_texts, truncation=True, max_length=128, batch_size=batch_size)
	results.extend(batch_results)

# Map model labels (1-5 stars) → sentiment + score
mapped = []
for r in results:
	label = r["label"]  # e.g. "1 star", "5 stars"
	score = r["score"]
	stars = int(label.split()[0])  # get number
	if stars <= 2:
		sentiment = "Negative"
	elif stars == 3:
		sentiment = "Neutral"
	else:
		sentiment = "Positive"
	mapped.append({"sentiment_score": score, "sentiment": sentiment})

# === 4. Build output dataframe ===
mapped_df = pd.DataFrame(mapped)
output_df = pd.concat([data[["textOriginal"]], mapped_df], axis=1)

# === 5. Save CSV ===
output_df.to_csv("comments_with_sentiment.csv", index=False)
print("Saved to comments_with_sentiment.csv")
print(output_df.head())
