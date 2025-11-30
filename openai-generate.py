#a simple method to generate the simplified names by using openai, needs a paid subscription to the service though so i didn't use it

import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import kagglehub
import json

# === CONFIGURATION ===
OPENAI_API_KEY = ""
INR_TO_GBP = 0.0085
OUTPUT_CSV = "generalized_gift_ideas.csv"
CACHE_FILE = "gpt_cache.json"

# === Setup ===
client = OpenAI(api_key=OPENAI_API_KEY)
tqdm.pandas()

# === Load GPT cache if exists ===
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        gpt_cache = json.load(f)
else:
    gpt_cache = {}

# === Get path to Kaggle dataset ===
print("📦 Downloading/loading dataset from Kaggle...")
path = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")

# === Load and combine CSVs ===
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

# === GPT generalization function ===
def generalize_product_name(name):
    if name in gpt_cache:
        return gpt_cache[name]  # Use cached result

    prompt = (
        f"Generalize the following product name into a broad gift idea type "
        f"(e.g., 'Bluetooth Speaker', 'Home Gym Set', 'Leather Journal'). "
        f"Only return the general gift type, nothing else.\n\nProduct: {name}\n\nGeneral Gift Idea:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a product name generalizer. Output only short, generalized gift idea categories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        generalized_name = response.choices[0].message.content.strip()
        gpt_cache[name] = generalized_name  # Cache result
        return generalized_name
    except Exception as e:
        print(f"❌ GPT failed on: {name[:40]}... | Error: {e}")
        gpt_cache[name] = "Unknown"
        return "Unknown"

# === Run GPT generalization ===
print("\n🧠 Generating gift ideas...")
df['gift_idea'] = df['name'].progress_apply(generalize_product_name)

# === Save updated cache ===
with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(gpt_cache, f, ensure_ascii=False, indent=2)

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

# === Clean output ===
grouped = grouped[grouped['gift_idea'].notnull()]
grouped = grouped[grouped['gift_idea'] != "Unknown"]

# === Save output ===
grouped.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Gift ideas saved to: {OUTPUT_CSV}")
