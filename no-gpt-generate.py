#this is a no-gpt version of the simplifying generator, it uses clustering and regex, didn't work very well though

#Clustering - "unsupervised machine learning technique" to group similar data into clusters 
#              (supervised learning uses labelled training data, it may know what the output should be)

#  1. First it cleans the names by removing unnecessary punctuation etc  (64)

#  2. Then it vectorizes the text (turns it into numbers, TF-IDF vectorization)  (75)
#     Each product becomes a coordinate in vector space

#  3. Words that are common are downweighted, words that are nique are upweighted  (75)
#     So "wireless mouse" and "bluetooth mouse" will be closer together than "dog leash"

#  4. Kmeans tries to split all the products into 100 clusters, grouping products with similar TF-IDF numbers  (81)
#     (Products like:"Bluetooth Speaker""Portable Wireless Speaker""Mini Bluetooth Soundbox" would likely go into the same cluster.)

#  5. Then it labels each cluster (gift idea) by finding product closest to cluster center   (102-116)
#     A cluster might be named “Wireless Speaker” even though it contains items like “JBL Soundbox” and “Bluetooth Speaker - Red”.

#This doesn't really work for the amazon products though because they are so varied and long which is why the cluster names weren't accurate



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
print("\n📊 Step 1: Vectorizing product names...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(tqdm(df['clean_name'], desc="Vectorizing"))

# === Cluster product names ===
print("\n🧠 Step 2: Clustering product names...")
n_clusters = 100  # adjust depending on how granular you want categories
kmeans = KMeans(n_clusters=n_clusters, random_state=42, verbose=1, n_init=10)  # verbose=1 shows progress
df['cluster'] = kmeans.fit_predict(X)


# === Find best representative label for each cluster ===
def clean_label(name):
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\[.*?\]", "", name)
    # Remove brand names (first word if it looks like a brand)
    name_parts = name.split()
    if len(name_parts) > 1:
        # Drop first word if it looks like a brand (capitalized word)
        if name_parts[0][0].isupper():
            name_parts = name_parts[1:]
    name = " ".join(name_parts)
    # Remove possessives and extra spaces
    name = re.sub(r"'s\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()


def get_cluster_labels_from_center(X, kmeans, df):
    labels = {}
    for cluster_id in tqdm(range(kmeans.n_clusters), desc="📌 Labelling clusters"):
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
