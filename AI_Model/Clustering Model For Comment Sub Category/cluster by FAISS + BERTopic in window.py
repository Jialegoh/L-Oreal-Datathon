import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import time
import faiss
import umap
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import stopwordsiso as stopwords_iso

print("🚀 Running FAISS + BERTopic-like clustering pipeline with improvements")

# ------------------------------
# 1. Load and merge datasets
# ------------------------------
print("Stage 1: Loading and merging datasets...")
df_spam = pd.read_csv("comments_evaluated.csv")  # textOriginal, is_spam
dfs = [pd.read_csv(f"commentVideoMerged{i}.csv") for i in range(1, 6)]
df_vid = pd.concat(dfs, ignore_index=True)

df_vid_clean = df_vid.drop_duplicates(subset=['textOriginal'])
df_spam_clean = df_spam.drop_duplicates(subset=['textOriginal'])
df = pd.merge(df_vid_clean, df_spam_clean, on="textOriginal", how="inner")

# ------------------------------
# 2. Filter out spam
# ------------------------------
print("Stage 2: Filtering out spam comments...")
df = df[df["is_spam"] == "No"].copy()
print(f"✅ After spam filter: {len(df)} rows")

# ------------------------------
# 2.5. Sample data for faster processing
# ------------------------------
SAMPLE_SIZE = 500000  # Adjust this based on your needs (50K is good for testing)
if len(df) > SAMPLE_SIZE:
    print(f"Stage 2.5: Sampling {SAMPLE_SIZE} comments for faster processing...")
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"✅ Using sample of {len(df)} comments")
else:
    print(f"✅ Dataset is small enough ({len(df)} rows), using full data")

# ------------------------------
# 3. Encode comments with memory-efficient approach
# ------------------------------
print("Stage 3: Encoding comments with multilingual model...")
texts = df["textOriginal"].astype(str).tolist()

print("📥 Encoding with multilingual SBERT (memory-efficient)...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Create memory-mapped array to store embeddings
embedding_dim = 384  # MiniLM-L12-v2 dimension
embeddings_file = "temp_embeddings.npy"
embeddings = np.memmap(embeddings_file, dtype='float32', mode='w+', shape=(len(texts), embedding_dim))

chunk_size = 1000
for i in tqdm(range(0, len(texts), chunk_size), desc="Encoding"):
    chunk_texts = texts[i:i+chunk_size]
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=False, batch_size=32)
    embeddings[i:i+len(chunk_embeddings)] = chunk_embeddings.astype('float32')
    
    # Force write to disk and free memory
    del chunk_embeddings
    embeddings.flush()

print(f"✅ Encoded {len(texts)} texts, stored in memory-mapped file")

# ------------------------------
# 4. Dimensionality reduction (UMAP) - Memory efficient
# ------------------------------
print("Stage 4: Reducing dimensions with UMAP...")
print("🔻 Reducing dimensions with UMAP...")

# For very large datasets, we can subsample for UMAP fitting
if len(texts) > 300000:
    print(f"📊 Large dataset ({len(texts)}), using subset for UMAP fitting...")
    sample_size = 200000
    sample_indices = np.random.choice(len(texts), size=sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]
    
    umap_model = umap.UMAP(
        n_neighbors=30,
        n_components=20,
        metric="cosine",
        random_state=42
    )
    umap_model.fit(sample_embeddings)
    del sample_embeddings  # Free memory
    
    # Transform full dataset in chunks
    embeddings_reduced_file = "temp_embeddings_reduced.npy"
    embeddings_reduced = np.memmap(embeddings_reduced_file, dtype='float32', mode='w+', shape=(len(texts), 20))
    
    chunk_size = 5000
    for i in tqdm(range(0, len(texts), chunk_size), desc="UMAP Transform"):
        end_idx = min(i + chunk_size, len(texts))
        chunk_reduced = umap_model.transform(embeddings[i:end_idx])
        embeddings_reduced[i:end_idx] = chunk_reduced.astype('float32')
        embeddings_reduced.flush()
    
else:
    umap_model = umap.UMAP(
        n_neighbors=30,
        n_components=20,
        metric="cosine",
        random_state=42
    )
    embeddings_reduced = umap_model.fit_transform(embeddings).astype("float32")

print(f"✅ Reduced {embeddings.shape[1]} → {embeddings_reduced.shape[1]} dims")

# ------------------------------
# 5. FAISS K-means clustering
# ------------------------------
print("Stage 5: Clustering with FAISS K-means...")
d = embeddings_reduced.shape[1]
n_clusters = min(150, max(20, len(texts) // 800))  # cap at 150 clusters
print(f"🔎 Clustering with FAISS (n_clusters={n_clusters})")

kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=False, gpu=False)

start = time.time()
kmeans.train(embeddings_reduced)
D, I = kmeans.index.search(embeddings_reduced, 1)
end = time.time()

df["cluster"] = I.flatten()
print(f"✅ Clustering done in {end-start:.1f}s, {n_clusters} clusters found")

# ------------------------------
# 6. Remove tiny clusters (< 50)
# ------------------------------
print("Stage 6: Removing tiny clusters...")
cluster_sizes = df["cluster"].value_counts()
valid_clusters = cluster_sizes[cluster_sizes >= 50].index  # more aggressive filtering

# Keep track of original row count before filtering
original_count = len(df)

# Create boolean mask for valid clusters
valid_mask = df["cluster"].isin(valid_clusters)

# Filter DataFrame and get corresponding positions
df = df[valid_mask].copy()
filtered_positions = np.where(valid_mask)[0]  # Get positions of True values in mask

# Filter embeddings_reduced to match the filtered DataFrame
print("🔄 Filtering embeddings to match DataFrame...")
embeddings_reduced_filtered = embeddings_reduced[filtered_positions]
embeddings_reduced = embeddings_reduced_filtered

# Reset DataFrame index to align with filtered embeddings
df = df.reset_index(drop=True)
print(f"✅ Removed tiny clusters. Remaining clusters: {len(valid_clusters)}")
print(f"✅ Filtered embeddings from {original_count} to {len(df)} rows")

# ------------------------------
# 7. Class-based TF-IDF (c-TF-IDF) with improved stopwords
# ------------------------------
print("Stage 7: Extracting semantic cluster labels with c-TF-IDF...")
docs_per_cluster = df.groupby("cluster")["textOriginal"].apply(lambda x: " ".join(x.dropna().astype(str))).tolist()
cluster_ids = df["cluster"].unique()

# Multilingual stopwords for YouTube comments (many languages)
print("🌍 Building multilingual stopwords list...")
all_stopwords = set()
try:
    # Merge stopwords from ALL supported languages
    for lang in stopwords_iso.langs():
        all_stopwords |= stopwords_iso.stopwords(lang)
    print(f"✅ Loaded stopwords for {len(stopwords_iso.langs())} languages")
except Exception as e:
    print(f"⚠️ Warning: Could not load multilingual stopwords: {e}")
    # Fallback to English only
    all_stopwords = set()

# Add custom "social media" stopwords
custom_stopwords = ["omg", "soo", "lol", "hahaha", "wow", "dang", 
                   "please", "thank", "thanks", "like", "really", 
                   "just", "good", "great", "amazing", "love", 
                   "nice", "beautiful", "awesome", "cool", "best",
                   "want", "need", "know", "think", "feel", "look",
                   "use", "make", "get", "go", "come", "see"]
all_stopwords |= set(custom_stopwords)

print(f"📝 Total stopwords: {len(all_stopwords)}")

vectorizer = CountVectorizer(
    stop_words=list(all_stopwords), 
    max_features=15000,
    token_pattern=r"(?u)\b\w\w+\b",  # ignore single characters
    min_df=2  # ignore words that appear in only 1 cluster
)

X_counts = vectorizer.fit_transform(docs_per_cluster)

transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=True)
c_tf_idf = transformer.fit_transform(X_counts)

terms = vectorizer.get_feature_names_out()
cluster_keywords = {}
for idx, cluster_id in enumerate(cluster_ids):
    row = c_tf_idf[idx].toarray()[0]
    top_indices = row.argsort()[::-1][:10]
    cluster_keywords[cluster_id] = [terms[i] for i in top_indices]

# ------------------------------
# 8. Merge highly similar topics (improved keyword overlap)
# ------------------------------
print("Stage 8: Checking for redundant clusters...")
print("🔄 Checking for redundant clusters with keyword overlap...")

# Method 1: Cosine similarity of TF-IDF vectors
similarity_matrix = cosine_similarity(c_tf_idf)
merged_clusters = {}
cosine_threshold = 0.7

# Method 2: Keyword overlap
keyword_threshold = 0.4  # 40% overlap of top keywords
for i in range(len(cluster_ids)):
    for j in range(i+1, len(cluster_ids)):
        # Check both cosine similarity and keyword overlap
        cosine_sim = similarity_matrix[i, j]
        
        keywords_i = set(cluster_keywords[cluster_ids[i]][:5])  # top 5 keywords
        keywords_j = set(cluster_keywords[cluster_ids[j]][:5])
        keyword_overlap = len(keywords_i & keywords_j) / 5.0
        
        if cosine_sim > cosine_threshold or keyword_overlap > keyword_threshold:
            merged_clusters.setdefault(cluster_ids[i], []).append(cluster_ids[j])
            print(f"🔗 Similar clusters {cluster_ids[i]} & {cluster_ids[j]}: cosine={cosine_sim:.3f}, overlap={keyword_overlap:.3f}")

print(f"✅ Found {len(merged_clusters)} groups of similar clusters")

# ------------------------------
# 8.5. Post-processing: Reassign low-confidence points
# ------------------------------
print("Stage 8.5: Post-processing reassignment of low-confidence points...")
print("🔄 Post-processing: reassigning low-confidence points...")

# 1. Compute centroids in reduced embedding space
cluster_ids = df["cluster"].unique()
centroids = {}
for cid in cluster_ids:
    mask = df["cluster"] == cid
    cluster_embeddings = embeddings_reduced[mask.values]  # Use boolean mask directly
    centroids[cid] = cluster_embeddings.mean(axis=0)

# 2. Convert to matrix for fast similarity checks
centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])
centroid_ids = list(cluster_ids)

# 3. Check similarity of each point to its assigned centroid
reassignments = 0
confidence_threshold = 0.35  # similarity below this is "low confidence"
margin_threshold = 0.1      # how much better new cluster must be

for i in range(len(df)):
    current_cluster = df.at[i, "cluster"]
    current_idx = centroid_ids.index(current_cluster)
    point_embedding = embeddings_reduced[i].reshape(1, -1)
    
    # Similarity to current centroid
    sim_to_own = cosine_similarity(
        point_embedding, centroid_matrix[current_idx].reshape(1, -1)
    )[0, 0]
    
    # If low confidence, try reassignment
    if sim_to_own < confidence_threshold:
        # Calculate similarity to all centroids
        sims = cosine_similarity(point_embedding, centroid_matrix)[0]
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        
        # Only reassign if new cluster is significantly better
        if best_sim > sim_to_own + margin_threshold:
            new_cluster = centroid_ids[best_idx]
            old_cluster = current_cluster
            df.at[i, "cluster"] = new_cluster
            reassignments += 1
            
            # Optionally print some reassignments for debugging
            if reassignments <= 10:  # Show first 10 reassignments
                comment_preview = df.at[i, "textOriginal"][:50] + "..."
                print(f"📝 Reassigned: '{comment_preview}' from cluster {old_cluster} → {new_cluster} (sim: {sim_to_own:.3f} → {best_sim:.3f})")

print(f"✅ Reassigned {reassignments} low-confidence comments to better clusters")

# Recalculate cluster keywords after reassignment
if reassignments > 0:
    print("🔄 Recalculating cluster keywords after reassignment...")
    docs_per_cluster = df.groupby("cluster")["textOriginal"].apply(lambda x: " ".join(x.dropna().astype(str))).tolist()
    cluster_ids = df["cluster"].unique()
    
    X_counts = vectorizer.fit_transform(docs_per_cluster)
    c_tf_idf = transformer.fit_transform(X_counts)
    terms = vectorizer.get_feature_names_out()
    
    cluster_keywords = {}
    for idx, cluster_id in enumerate(cluster_ids):
        row = c_tf_idf[idx].toarray()[0]
        top_indices = row.argsort()[::-1][:10]
        cluster_keywords[cluster_id] = [terms[i] for i in top_indices]

# ------------------------------
# 8.7. Optional: Actually merge highly similar clusters
# ------------------------------
if len(merged_clusters) > 0:
    print("Stage 8.7: Actually merging highly similar clusters...")
    merge_count = 0
    
    for main_cluster, similar_clusters in merged_clusters.items():
        # Only merge if both clusters still exist (haven't been merged already)
        if main_cluster in df["cluster"].values:
            for similar_cluster in similar_clusters:
                if similar_cluster in df["cluster"].values:
                    # Merge similar_cluster into main_cluster
                    df.loc[df["cluster"] == similar_cluster, "cluster"] = main_cluster
                    merge_count += 1
                    print(f"🔗 Merged cluster {similar_cluster} → {main_cluster}")
    
    if merge_count > 0:
        print(f"✅ Merged {merge_count} redundant clusters")
        
        # Recalculate final cluster keywords after merging
        print("🔄 Recalculating final cluster keywords after merging...")
        docs_per_cluster = df.groupby("cluster")["textOriginal"].apply(lambda x: " ".join(x.dropna().astype(str))).tolist()
        cluster_ids = df["cluster"].unique()
        
        X_counts = vectorizer.fit_transform(docs_per_cluster)
        c_tf_idf = transformer.fit_transform(X_counts)
        terms = vectorizer.get_feature_names_out()
        
        cluster_keywords = {}
        for idx, cluster_id in enumerate(cluster_ids):
            row = c_tf_idf[idx].toarray()[0]
            top_indices = row.argsort()[::-1][:10]
            cluster_keywords[cluster_id] = [terms[i] for i in top_indices]

# ------------------------------
# 9. Save results
# ------------------------------
print("Stage 9: Saving results...")
df.to_csv("clustered_comments_faiss_bertopic_improved.csv", index=False)
print("💾 Results saved to clustered_comments_faiss_bertopic_improved.csv")

print("\n" + "="*60)
print("🔥 FAISS + BERTopic-like CLUSTERING COMPLETE!")
print("="*60)
print(f"Final number of clusters: {len(cluster_ids)} (after filtering)")
print("Top semantic labels per cluster:")
for c, kw in cluster_keywords.items():
    print(f"Cluster {c}: {kw}")

# ------------------------------
# 10. Cleanup temporary files
# ------------------------------
import os
try:
    if os.path.exists("temp_embeddings.npy"):
        os.remove("temp_embeddings.npy")
    if os.path.exists("temp_embeddings_reduced.npy"):
        os.remove("temp_embeddings_reduced.npy")
    print("🧹 Cleaned up temporary files")
except:
    print("⚠️ Could not clean up temporary files (they may still be in use)")
