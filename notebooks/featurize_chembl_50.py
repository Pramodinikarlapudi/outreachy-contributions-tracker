import pandas as pd
import numpy as np
from ersilia import ErsiliaModel

df = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_chembl_50.csv")
hERG_smiles = df['Smiles'].tolist()
hERG_labels = df['LABEL (Y)'].tolist()

model = ErsiliaModel("eos4u6p")
model.serve()

print("Valid SMILES:", len(hERG_smiles))

# hERG_smiles = [smi for smi in hERG_all["Drug"].tolist() if isinstance(smi, str) and smi]  # Only valid strings
# hERG_labels = [lbl for smi, lbl in zip(hERG_all["Drug"], hERG_all["Y"]) if isinstance(smi, str) and smi]  # Match labels


# Featurizing with batching (Didn't run with out batching -- Huge data)

def batch_run(smiles_list, batch_size=10):
    features = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        batch_features = [item['output']['outcome'] for item in model.run(batch)]
        features.extend(batch_features)
        print(f"Processed {len(features)} of {len(smiles_list)}")
    return np.array(features)

hERG_features = batch_run(hERG_smiles)

# Saving hERG
hERG_features_df = pd.DataFrame(hERG_features, columns=[f'SIG{i+1}' for i in range(3200)])
hERG_features_df['SMILES'] = hERG_smiles
hERG_features_df['Label'] = hERG_labels
hERG_features_df.to_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_chembl_features.csv", index=False)

print("hERG features shape:", hERG_features.shape)