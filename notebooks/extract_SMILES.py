import pandas as pd

# Load datasets
train = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_train.csv")
valid = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_valid.csv")
test = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_test.csv")

# Combine and take first 10
all_smiles = pd.concat([train['Drug'], valid['Drug'], test['Drug']])
ten_smiles = all_smiles.head(10)
ten_smiles.to_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_10_smiles.txt", index=False, header=False)
print("Saved 10 SMILES to hERG_10_smiles.txt")
print("First 10 SMILES:", ten_smiles.tolist())