#this was used to generate a small training file of simplified names for the ai


import os
import re
import json
import pandas as pd
import kagglehub

APPROVED_KEYWORDS_FILE = "approved_keywords.json"
STOPWORDS_FILE = "stopwords.json"

print("📦 Downloading dataset...")
path = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")

print("📂 Loading CSV files...")
dataframes = []
for file in os.listdir(path):
    if file.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(path, file))
            dataframes.append(df)
            print(f"✅ Loaded: {file}")
        except Exception as e:
            print(f"⚠️ Could not load {file}: {e}")

df = pd.concat(dataframes, ignore_index=True)
df = df[['name', 'main_category']].dropna()

# Load approved keywords
try:
    with open(APPROVED_KEYWORDS_FILE, 'r') as f:
        approved_keywords = json.load(f)
    print(f"🔄 Loaded {len(approved_keywords)} approved keywords.")
except FileNotFoundError:
    print(f"❌ No approved keywords found at {APPROVED_KEYWORDS_FILE}. Exiting.")
    exit(1)

# Load stopwords
try:
    with open(STOPWORDS_FILE, 'r') as f:
        stopwords = set(json.load(f))
    print(f"🔄 Loaded {len(stopwords)} stopwords.")
except FileNotFoundError:
    print(f"⚠️ No stopwords file found at {STOPWORDS_FILE}. Continuing with empty set.")
    stopwords = set()

def remove_stopwords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    filtered = [w for w in words if w not in stopwords]
    return " ".join(filtered).strip()

def simplify_name(name):
    name_lower = name.lower()
    # Try matching approved keyword
    for kw in sorted(approved_keywords.keys(), key=len, reverse=True):
        if kw in stopwords:
            continue  # Skip stopwords completely
        if re.search(rf"\b{re.escape(kw)}\b", name_lower):
            return approved_keywords[kw]
    # If no approved keyword matched, check if all words are stopwords
    words = re.findall(r'\b\w+\b', name_lower)
    if all(word in stopwords for word in words):
        return None  # Skip this row
    # Remove stopwords from fallback name
    cleaned = remove_stopwords(name)
    return cleaned.split(",")[0].split("|")[0].strip()

df['simplified_idea'] = df['name'].apply(simplify_name)

# Remove rows where simplified_idea is None or empty
df = df[df['simplified_idea'].str.strip().astype(bool)]

sample_size = 15
balanced_dfs = []
for cat, group_df in df.groupby('main_category'):
    balanced_dfs.append(group_df.sample(n=min(sample_size, len(group_df)), random_state=42))

balanced_df = pd.concat(balanced_dfs)

balanced_df.rename(columns={'name': 'product_name'}, inplace=True)
balanced_df[['product_name', 'simplified_idea']].to_csv("train.csv", index=False)

print(f"\n✅ Balanced dataset saved to train.csv with {len(balanced_df)} rows (stopwords removed from simplified ideas)")
