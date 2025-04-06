import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

train = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_train.csv")
valid = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_valid.csv")
test = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_test.csv")

data = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_Morgan_fingerprints_features.csv")
X = data.drop(columns=['SMILES', 'Label'])
y = data['Label']

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='skyblue', label='0 (Non-blockers)', alpha=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='salmon', label='1 (Blockers)', alpha=0.5)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('hERG PCA (eos4wt0 Features)')
plt.legend()
plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/Morgan Finger prints_hERG_pca_labels.png')
plt.close()
