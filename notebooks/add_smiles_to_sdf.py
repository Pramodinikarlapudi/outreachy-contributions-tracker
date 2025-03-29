import pandas as pd

# Loading SMILES
train = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_train.csv")
valid = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_valid.csv")
test = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_test.csv")
all_smiles = pd.concat([train['Drug'], valid['Drug'], test['Drug']]).head(10)

# Converting them with titles
with open("/mnt/d/outreachy-contributions-tracker/data/hERG_10_smiles.txt", "w") as f:
    for i, smiles in enumerate(all_smiles, 1):
        f.write(f"{smiles} Molecule_{i}\n")  # SMILES + name

print("Saved SMILES with titles to hERG_10_smiles.txt")