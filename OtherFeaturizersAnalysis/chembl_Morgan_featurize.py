import pandas as pd
import numpy as np
from ersilia import ErsiliaModel

# Load hERG
df = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_chembl_50.csv")
hERG_smiles = df['Smiles'].tolist()
hERG_labels = df['LABEL (Y)'].tolist()

# Load model
model = ErsiliaModel("eos4wt0")
model.serve()

# Featurize with batching
def batch_run(smiles_list, batch_size=10):
    features = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        batch_features = [item['output']['outcome'] for item in model.run(batch)]
        features.extend(batch_features)
        print(f"Processed {len(features)} of {len(smiles_list)}")
    return np.array(features)

hERG_features = batch_run(hERG_smiles, batch_size=10)

# Save features
feature_columns = [f'feature_{i}' for i in range(hERG_features.shape[1])]
hERG_features_df = pd.DataFrame(hERG_features, columns=feature_columns)
hERG_features_df['SMILES'] = hERG_smiles
hERG_features_df['Label'] = hERG_smiles
hERG_features_df.to_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_Chembl_Morgan_fingerprints_features.csv", index=False)

print("hERG features shape:", hERG_features.shape)
print("Features saved!")