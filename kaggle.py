import kagglehub

# Download latest version
path = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")

print(path)

import os

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
print("CSV files:", csv_files)
