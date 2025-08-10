import os
import re
import pandas as pd
from tqdm import tqdm
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# === CONFIGURATION ===
INR_TO_GBP = 0.0085
OUTPUT_CSV = "generalized_gift_ideas.csv"

# === Download dataset from Kaggle ===
print("📦 Downloading/loading dataset from Kaggle...")
path = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")

# === Load and combine all CSV files ===
print("📂 Loading CSV files...")
all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
dataframes = []

for file in all_files:
    try:
        df = pd.read_csv(os.path.join(path, file))
        dataframes.append(df)
        print(f"✅ Loaded: {file}")
    except Exception as e:
        print(f"⚠️ Could not load {file}: {e}")

df = pd.concat(dataframes, ignore_index=True)

# === Keep only needed columns ===
needed_cols = ['name', 'main_category', 'sub_category', 'ratings', 'actual_price']
df = df[[c for c in needed_cols if c in df.columns]]

print(f"\n🧾 Total products loaded: {len(df)}")
print("🧪 Sample:", df['name'].iloc[0])

# === Clean product names ===
def clean_text(text):
    text = str(text)
    text = re.sub(r"\(.*?\)", "", text)  # remove bracketed text
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove special chars
    return text.lower().strip()

df['clean_name'] = df['name'].apply(clean_text)

# === Vectorize product names ===
print("\n🔠 Vectorizing product names...")
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_name'])

# === Cluster products ===
print("🤖 Clustering products...")
n_clusters = 100  # adjust depending on how granular you want categories
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# === Find best representative label for each cluster ===
def clean_label(name):
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\[.*?\]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()

def get_cluster_labels_from_center(X, kmeans, df):
    labels = {}
    for cluster_id in range(kmeans.n_clusters):
        idxs = df[df['cluster'] == cluster_id].index
        if len(idxs) == 0:
            labels[cluster_id] = f"Cluster {cluster_id}"
            continue
        center = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        closest_idx, _ = pairwise_distances_argmin_min(center, X[idxs])
        representative_name = df.loc[idxs[closest_idx[0]], 'name']
        labels[cluster_id] = clean_label(representative_name)
    return labels

cluster_labels = get_cluster_labels_from_center(X, kmeans, df)
df['gift_idea'] = df['cluster'].map(cluster_labels)

# === Convert INR → GBP ===
def inr_to_gbp(value):
    try:
        value = str(value).replace(',', '').replace('₹', '').strip()
        return round(float(value) * INR_TO_GBP, 2)
    except:
        return None

df['price_gbp'] = df['actual_price'].apply(inr_to_gbp)
df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')

# === Group by gift idea ===
grouped = df.groupby('gift_idea').agg({
    'price_gbp': 'mean',
    'ratings': 'mean',
    'main_category': lambda x: list(set(x.dropna())),
    'sub_category': lambda x: list(set(x.dropna())),
    'name': 'count'
}).reset_index()

grouped.rename(columns={
    'price_gbp': 'average_price_gbp',
    'ratings': 'average_rating',
    'main_category': 'main_categories',
    'sub_category': 'sub_categories',
    'name': 'num_products'
}, inplace=True)

# === Save output ===
grouped.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Gift ideas saved to: {OUTPUT_CSV}")
