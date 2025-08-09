import os
import pandas as pd
import openai
from tqdm import tqdm

# === CONFIGURATION ===
OPENAI_API_KEY = "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"  # ← Replace with your real key
INR_TO_GBP = 0.0085
OUTPUT_CSV = "generalized_gift_ideas.csv"

# === Set up OpenAI API ===
openai.api_key = OPENAI_API_KEY
tqdm.pandas()

# === Get path to Kaggle dataset (already downloaded using kagglehub) ===
import kagglehub
path = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")

# === Load and combine all CSVs ===
print("📦 Loading all product CSVs...")
all_files = [f for f in os.listdir(path) if f.endswith('.csv')]

dataframes = []
for file in all_files:
    file_path = os.path.join(path, file)
    try:
        df = pd.read_csv(file_path)
        dataframes.append(df)
        print(f"✅ Loaded: {file}")
    except Exception as e:
        print(f"⚠️ Failed to load {file}: {e}")

df = pd.concat(dataframes, ignore_index=True)

# === Clean and preview ===
expected_columns = ['name', 'main_category', 'sub_category', 'ratings', 'discount_price']
df = df[[col for col in expected_columns if col in df.columns]]

print(f"\n🧾 Total products loaded: {len(df)}")
print("🧪 Sample product:", df['name'].iloc[0])

# === GPT-based generalization ===
def generalize_product_name(name):
    prompt = f"Generalize the following product name into a broad gift idea type (e.g., 'Bluetooth Speaker', 'Home Gym Set', 'Leather Journal', etc.):\n\nProduct: {name}\n\nGeneral Gift Idea:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a helpful assistant that generalizes product names into gift idea types."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.3
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"❌ GPT failed on: {name[:40]}... | Error: {e}")
        return "Unknown"

print("\n🧠 Generating gift idea types from product names...")
df['gift_idea'] = df['name'].progress_apply(generalize_product_name)

# === Convert price to GBP ===
def inr_to_gbp(value):
    try:
        value = str(value).replace(',', '').replace('₹', '').strip()
        return round(float(value) * INR_TO_GBP, 2)
    except:
        return None

df['price_gbp'] = df['discount_price'].apply(inr_to_gbp)
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
