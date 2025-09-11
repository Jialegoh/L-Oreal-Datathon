import pandas as pd
import re
import ast
from tqdm import tqdm

# Enable pandas progress bar
tqdm.pandas()

# Load the CSV
df = pd.read_csv("clustered_comments_faiss_bertopic_improved.csv")

# Drop the old cluster column if exists
if "cluster" in df.columns:
    df = df.drop(columns=["cluster"])

# Function to parse cluster.txt file
def parse_cluster_file(filename):
    clusters = {}
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line and ':' in line:
                    # Split on the first colon to separate cluster name and keywords
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        # Extract cluster name (remove quotes)
                        cluster_name = parts[0].strip().strip("'\"")
                        
                        # Parse the list of keywords using ast.literal_eval
                        try:
                            keywords = ast.literal_eval(parts[1].strip())
                            if isinstance(keywords, list):
                                clusters[cluster_name] = keywords
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing line: {line}")
                            print(f"Error: {e}")
                            continue
        
        print(f"✅ Successfully loaded {len(clusters)} clusters from {filename}")
        return clusters
    
    except FileNotFoundError:
        print(f"❌ Error: {filename} not found!")
        return {}
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return {}

# Load clusters from cluster.txt
clusters = parse_cluster_file("cluster.txt")

if not clusters:
    print("❌ No clusters loaded. Exiting...")
    exit(1)

# Display loaded clusters for verification
print(f"\n📋 Loaded clusters:")
for cluster_name, keywords in clusters.items():
    print(f"  - {cluster_name}: {len(keywords)} keywords")

# Compile regex patterns for efficiency
print(f"\n🔧 Compiling regex patterns for {len(clusters)} clusters...")
cluster_patterns = {
    name: re.compile("|".join([re.escape(word.lower()) for word in words]), re.IGNORECASE)
    for name, words in clusters.items()
}

# Function to assign cluster
def assign_cluster(text):
    text_l = str(text).lower()
    for cluster_name, pattern in cluster_patterns.items():
        if pattern.search(text_l):
            return cluster_name
    return "General / Miscellaneous"

# Apply cluster assignment with progress bar
print(f"\n🔄 Assigning clusters to {len(df):,} comments...")
# Set the progress bar description using tqdm.pandas()
tqdm.pandas(desc="Clustering comments")
df["new_cluster"] = df["textOriginal"].progress_apply(assign_cluster)

# Normalize cluster names (remove quotes, curly quotes, extra spaces)
print(f"\n🔧 Normalizing cluster names...")
df["new_cluster"] = df["new_cluster"].astype(str).str.strip()
df["new_cluster"] = df["new_cluster"].str.replace(r"['']", "'", regex=True)  # replace curly quotes with normal '
df["new_cluster"] = df["new_cluster"].str.replace(r"^'+|'+$", "", regex=True)  # strip leading/trailing quotes

# Debug: Check for any remaining quote issues in spammy clusters
print(f"\n🔍 Checking for quote issues in spammy cluster names:")
for c in df["new_cluster"].unique():
    if any(spam_word in c.lower() for spam_word in ["spam", "ads", "random", "troll", "promotion"]):
        print(f"  Found: [{c}]")

# Display cluster distribution
cluster_counts = df["new_cluster"].value_counts()
print(f"\n📊 Cluster distribution:")
for cluster, count in cluster_counts.head(10).items():
    print(f"  - {cluster}: {count:,} comments")

if len(cluster_counts) > 10:
    print(f"  ... and {len(cluster_counts) - 10} more clusters")

# Define spammy clusters
spammy_clusters = {
    "Spam / Troll Comments",
    "Ads / Promotions / Trends", 
    "Random Words / Miscellaneous"
}

# Show current spam counts before update
print(f"\n📊 Current spam status:")
current_spam_counts = df["is_spam"].value_counts()
print(f"  - Spam: {current_spam_counts.get('Yes', 0):,}")
print(f"  - Not Spam: {current_spam_counts.get('No', 0):,}")

# Update is_spam based on new_cluster
print(f"\n⚠️ Updating is_spam for spammy clusters...")
tqdm.pandas(desc="Updating spam status")
df["is_spam"] = df.progress_apply(
    lambda row: "Yes" if (row["new_cluster"] in spammy_clusters or row["is_spam"] == "Yes") else "No",
    axis=1
)

# Show updated spam counts
print(f"\n📊 Spam status after update:")
updated_spam_counts = df["is_spam"].value_counts()
print(f"  - Spam: {updated_spam_counts.get('Yes', 0):,}")
print(f"  - Not Spam: {updated_spam_counts.get('No', 0):,}")

# Show what changed
newly_marked_spam = updated_spam_counts.get('Yes', 0) - current_spam_counts.get('Yes', 0)
print(f"  - Newly marked as spam: {newly_marked_spam:,}")

# Show breakdown by spammy cluster
print(f"\n🔍 Breakdown of spammy clusters:")
for cluster in spammy_clusters:
    count = cluster_counts.get(cluster, 0)
    if count > 0:
        print(f"  - {cluster}: {count:,} comments")

# Save to new CSV
output_filename = "clustered_comments_reassigned.csv"
print(f"\n💾 Saving results to {output_filename}...")
df.to_csv(output_filename, index=False)

print(f"\n✅ Reclustering complete. Saved as {output_filename}")
print(f"📈 Total comments: {len(df):,}")
print(f"🏷️ Total clusters assigned: {len(cluster_counts)}")
print(f"❓ General/Miscellaneous comments: {cluster_counts.get('General / Miscellaneous', 0):,}")
print(f"🚫 Total spam comments: {updated_spam_counts.get('Yes', 0):,}")
