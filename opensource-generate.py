# === This only works with python 3.12 or less ===
#This is the one that works! Uses flan-t5 opensource nlp trained with my data


import os
import pandas as pd
from tqdm import tqdm
import kagglehub
import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# === CONFIGURATION ===
#MODEL_NAME = "t5-small"
MODEL_NAME = "flan_t5_gift_idea_finetuned" 
CACHE_FILE = "simplify_cache.json"
OUTPUT_CSV = "generalized_gift_ideas.csv"
INR_TO_GBP = 0.0085

# === Setup tokenizer and model ===
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

tqdm.pandas()

# === Load cache if exists ===
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        simplify_cache = json.load(f)
else:
    simplify_cache = {}

# === Load dataset ===
print("📦 Downloading/loading dataset from Kaggle...")
path = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")

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

# After concatenating all dataframes:
df = pd.concat(dataframes, ignore_index=True)

# --- SAMPLE THE DATASET ---
sample_size = 100  # or set a fraction like 0.1 for 10%
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
print(f"Sampled dataset size: {len(df)}")

# Then continue with your existing column filtering etc.
needed_cols = ['name', 'main_category', 'sub_category', 'ratings', 'actual_price']
df = df[[c for c in needed_cols if c in df.columns]]


print(f"\n🧾 Total products loaded: {len(df)}")
print("🧪 Sample:", df['name'].iloc[0])

# === Simplify function using model inference ===
def generalize_product_name(name):
    if name in simplify_cache:
        return simplify_cache[name]

    # Prepare input for T5: prefix + text (you can customize prompt)
    input_text = f"simplify: {name}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)

    # Generate output sequence
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=15,  # gift ideas are short
            num_beams=5,
            early_stopping=True
        )
    generalized_name = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    simplify_cache[name] = generalized_name
    return generalized_name

# === Run generalization on dataset ===
print("\n🧠 Generating gift ideas...")
df['gift_idea'] = df['name'].progress_apply(generalize_product_name)

# === Save cache ===
with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(simplify_cache, f, ensure_ascii=False, indent=2)

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
    'name': lambda x: list(x)  # Save all original product names
}).reset_index()

# Rename columns
grouped.rename(columns={
    'price_gbp': 'average_price_gbp',
    'ratings': 'average_rating',
    'main_category': 'main_categories',
    'sub_category': 'sub_categories',
    'name': 'original_product_names',  # <-- updated
}, inplace=True)

# Remove empty or unknown gift ideas
grouped = grouped[grouped['gift_idea'].notnull()]
grouped = grouped[grouped['gift_idea'] != "Unknown"]

grouped.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Gift ideas saved to: {OUTPUT_CSV}")