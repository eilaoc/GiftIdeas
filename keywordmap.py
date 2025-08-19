import os
import re
import json
import pandas as pd
from collections import Counter
from itertools import islice
import kagglehub

APPROVED_KEYWORDS_FILE = "approved_keywords.json"
STOPWORDS_FILE = "stopwords.json"

# Load previously denied stopwords if available
try:
    with open(STOPWORDS_FILE, 'r') as f:
        denied_stopwords = set(json.load(f))
    print(f"🔄 Loaded {len(denied_stopwords)} previously denied keywords as stopwords.")
except FileNotFoundError:
    denied_stopwords = set()




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

keyword_map = {
    "umbrella": "Umbrella",
    "headphone": "Headphones",
    "earbuds": "Earbuds",
    "trolley": "Trolley Bag",
    "backpack": "Backpack",
    "necklace": "Necklace",
    "watch": "Watch",
    "shoes": "Shoes",
    "sofa": "Sofa",
    "mixer": "Mixer Grinder",
    "camera": "Camera",
    "microwave": "Microwave",
    "guitar": "Guitar",
    "football": "Football",
    "basketball": "Basketball",
    "t-shirt": "T-Shirt",
    "shirt": "Shirt",
    "jacket": "Jacket",
    "earring": "Earrings",
    "ring": "Ring"
}

# Load previously approved keywords if available
try:
    with open(APPROVED_KEYWORDS_FILE, 'r') as f:
        approved_keywords = json.load(f)
    print(f"🔄 Loaded {len(approved_keywords)} previously approved keywords.")
except FileNotFoundError:
    approved_keywords = {}

current_keywords = set(list(keyword_map.keys()) + list(approved_keywords.keys()))

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

all_names = df['name'].dropna()

all_words = []
for name in all_names:
    all_words.extend(tokenize(name))

def get_bigrams(words):
    return zip(words, islice(words, 1, None))

all_bigrams = []
for name in all_names:
    words = tokenize(name)
    all_bigrams.extend([' '.join(bigram) for bigram in get_bigrams(words)])

word_counts = Counter(all_words)
bigram_counts = Counter(all_bigrams)

stopwords = set([
    # demographics / gender
    'women', 'men', 'mens', 'girls', 'boys', 'unisex', 'adult', 'baby', 'kids',
    # colors
    'black', 'white', 'blue', 'yellow', 'brown', 'green', 'grey', 'red', 'pink',
    'orange', 'purple', 'violet', 'gold', 'silver', 'rose',
    # sizes / fits / styles
    'fit', 'regular', 'slim', 'casual', 'formal', 'stylish', 'free', 'mini', 'full', 'half',
    # common adjectives / descriptors
    'new', 'best', 'latest', 'soft', 'portable', 'light', 'solid', 'round', 'high',
    'compatible', 'wireless', 'bluetooth', 'digital', 'analog',
    # generic product words (to ignore these because too generic or often modifiers)
    'pack', 'set', 'box', 'price', 'offer', 'quality', 'made', 'material', 'case',
    'sale', 'product', 'wear', 'combo',
    # brands / model words (common ones)
    'amazon', 'brand', 'van', 'heusen', 'polo', 'assn', 'boost', 'bass', 'remote',
    # suffixes for verb/adjective forms to avoid
    'ing', 'ed', 'ly', 'er', 'or', 'ion', 'ive',
    # other generic terms
    'top', 'piece', 'strap', 'dial', 'sound', 'inch', 'sleeve',
    # common stopwords
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
])

stopwords.update(denied_stopwords)


def looks_like_noun(word):
    bad_suffixes = ('ing', 'ed', 'ly', 'er', 'or', 'ion', 'ive')
    if any(word.endswith(suf) for suf in bad_suffixes):
        return False
    if word in stopwords:
        return False
    return True

def phrase_is_clean(phrase):
    return all(looks_like_noun(w) and w not in stopwords for w in phrase.split())

candidate_words = [
    (word, count) for word, count in word_counts.items()
    if word not in current_keywords and len(word) > 2 and looks_like_noun(word)
]

candidate_bigrams = [
    (phrase, count) for phrase, count in bigram_counts.items()
    if phrase_is_clean(phrase) and phrase not in current_keywords
]

candidate_words.sort(key=lambda x: x[1], reverse=True)
candidate_bigrams.sort(key=lambda x: x[1], reverse=True)

new_denied_keywords = set()


print("\nInteractive keyword approval:")
def approve_candidates(candidates, is_bigram=False):
    global approved_keywords, new_denied_keywords
    for phrase, count in candidates:
        if phrase in current_keywords or phrase in stopwords:
            continue  # skip already known or denied words

        display = phrase.title() if is_bigram else phrase.capitalize()
        while True:
            ans = input(f"Approve '{display}' (count={count})? (y/n/q): ").strip().lower()
            if ans == 'y':
                approved_keywords[phrase] = display
                break
            elif ans == 'n':
                new_denied_keywords.add(phrase)  # <--- Add denied keyword here
                break
            elif ans == 'q':
                print("Stopping approval process early.")
                return False
            else:
                print("Please enter y (yes), n (no), or q (quit).")
    return True



print("\nApprove single-word candidates:")
if not approve_candidates(candidate_words[:50], is_bigram=False):
    pass

print("\nApprove bigram candidates:")
if not approve_candidates(candidate_bigrams[:30], is_bigram=True):
    pass

print(f"\n✅ Approved keywords so far: {len(approved_keywords)}")

# Save approved keywords
with open(APPROVED_KEYWORDS_FILE, 'w') as f:
    json.dump(approved_keywords, f, indent=2)

# Save denied keywords to stopwords file
denied_stopwords.update(new_denied_keywords)
with open(STOPWORDS_FILE, 'w') as f:
    json.dump(sorted(denied_stopwords), f, indent=2)

print(f"💾 Saved approved keywords to {APPROVED_KEYWORDS_FILE}")
print(f"💾 Saved denied keywords (stopwords) to {STOPWORDS_FILE}")