#This is the code to put the subcategories into a graph joining the similar categories
#It first extracts the subcategories
#Then generate embeddings using sentence tranformers for the subcategories (represent the word with a number in high dimensional space, similar to clustering)
#Then calculate the cosine similarity between embeddings, similar numbers are clustered together
    #Example (simplified, 3 dimensions instead of 768)
        #"Men's Clothing" → [0.8, 0.1, 0.6]
        #"Women's Clothing" → [0.75, 0.15, 0.58]
        #"Makeup" → [0.1, 0.9, 0.2]
    #Now we can compute the cosine similarity between them:
        #similarity(a,b)= a.b/||a||b||
        #Clothing vs Clothing → 0.98 (very close)
        #Clothing vs Makeup → 0.34 (not close)
    #If the similarity is over the threshold, add an edge
#Then we create the graph and add the nodes
#Add all the edges using an adjacency matrix
#Then visualize the graph with networkx and plt






import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# === Load CSV ===
df = pd.read_csv("generalized_gift_ideas.csv")

# === Extract unique subcategories ===
print("🔍 Extracting unique subcategories...")
subcategories = set()
for sublist in tqdm(df["sub_categories"].dropna()):
    for subcat in eval(sublist):  # assumes list-like strings (e.g. "['A', 'B']")
        subcategories.add(subcat.strip())

subcategories = sorted(list(subcategories))
print(f"✅ Found {len(subcategories)} unique subcategories.")

# === Generate embeddings ===
print("🧠 Generating embeddings for subcategories...")
model = SentenceTransformer('all-mpnet-base-v2')


# Use tqdm with list comprehension
embeddings = list(tqdm(model.encode(subcategories, convert_to_tensor=True), total=len(subcategories)))

# === Compute cosine similarity matrix ===
print("📐 Computing similarity matrix...")
similarity_matrix = cosine_similarity([e.cpu().numpy() for e in embeddings])

# === Build similarity graph ===
print("🕸️ Building graph...")
G = nx.Graph()

# Add nodes
for subcat in subcategories:
    G.add_node(subcat)

# Add edges based on similarity threshold
SIMILARITY_THRESHOLD = 0.45  # adjust as needed
num_edges = 0
for i in tqdm(range(len(subcategories))):
    for j in range(i + 1, len(subcategories)):
        sim = similarity_matrix[i][j]
        if sim > SIMILARITY_THRESHOLD:
            G.add_edge(subcategories[i], subcategories[j], weight=sim)
            num_edges += 1

print(f"✅ Graph created with {len(G.nodes)} nodes and {num_edges} edges.")

# === Visualize graph ===
print("📊 Visualizing graph...")
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=0.5, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')

# Draw edges (only those above threshold)
nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title("Subcategory Similarity Graph", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
