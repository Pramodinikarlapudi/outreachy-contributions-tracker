import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

tdc_data = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_Morgan_fingerprints_features.csv")
chembl_data = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_Chembl_Morgan_fingerprints_features.csv")

signature_columns = [f'feature_{i}' for i in range(2048)]

X_tdc = tdc_data[signature_columns].values
X_chembl = chembl_data[signature_columns].values

X_combined = np.vstack((X_tdc, X_chembl))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)

n_tdc = X_tdc.shape[0] 
X_tdc_pca = X_pca[:n_tdc]
X_chembl_pca = X_pca[n_tdc:]


plt.figure(figsize=(10, 6))
plt.scatter(X_tdc_pca[:, 0], X_tdc_pca[:, 1], c='blue', label='TDC Training Set', alpha=0.5)
plt.scatter(X_chembl_pca[:, 0], X_chembl_pca[:, 1], c='orange', label='ChEMBL Test Set', alpha=0.5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of TDC Training Set vs ChEMBL Test Set (Morgan Fingerprints)')
plt.legend()
plt.grid(True)


plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/PCA_TDC_vs_ChEMBL.png')
plt.close()

print("PCA plot saved to /mnt/d/outreachy-contributions-tracker/figures/PCA_TDC_vs_ChEMBL.png")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")