import pandas as pd
import numpy as np
from ersilia import ErsiliaModel

hERG_train = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_train.csv")
hERG_valid = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_valid.csv")
hERG_test = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_test.csv")
hERG_all = pd.concat([hERG_train, hERG_valid, hERG_test])
hERG_smiles = hERG_all["Drug"].tolist()

model = ErsiliaModel("eos2db3")
model.serve()

def batch_run(smiles_list, batch_size=10):
    features = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        batch_features = [item['output']['outcome'] for item in model.run(batch)]
        features.extend(batch_features)
        print(f"Processed {len(features)} of {len(smiles_list)}")
    return np.array(features)

hERG_features = batch_run(hERG_smiles, batch_size=10)


hERG_features_df = pd.DataFrame(hERG_features, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'UMAP1', 'UMAP2', 'tSNE1', 'tSNE2'])
hERG_features_df['SMILES'] = hERG_smiles
hERG_features_df['Label'] = hERG_all['Y'].values
hERG_features_df.to_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_chemDiv_features.csv", index=False)

print("hERG features shape:", hERG_features.shape)
print("Features saved!")
