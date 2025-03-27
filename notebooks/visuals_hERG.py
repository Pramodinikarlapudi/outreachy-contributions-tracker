
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

train = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_train.csv")
valid = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_valid.csv")
test = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_test.csv")

print("Train - ", len(train), "0s:", (train['Y'] == 0).sum(), "1s:", (train['Y']==1).sum())
print("Valid - ", len(valid), "0s:", (valid['Y'] == 0).sum(), "1s:", (valid['Y']==1).sum())
print("Test - ", len(test), "0s:", (test['Y'] == 0).sum(), "1s:", (test['Y']==1).sum())

sets = ['Train', 'Valid', 'Test']
zeros = [(train['Y'] == 0).sum(), (valid['Y'] == 0).sum(), (test['Y'] == 0).sum()]
ones = [(train['Y'] == 1).sum(), (valid['Y'] == 1).sum(), (test['Y'] == 1).sum()]

plt.bar(sets,zeros,label='0 (Non-blockers)', color='skyblue')
plt.bar(sets,ones,bottom=zeros, label='1 (Blockers)', color='salmon')

plt.xlabel('Dataset')
plt.ylabel('Count')

plt.title('hERG Label Distribution')

plt.legend()
plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/hERG_label_dist.png')

plt.close()

# Featurized data

data = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_ccsign_features.csv")
X = data.drop(columns=['SMILES', 'Label'])
y = data['Label']

# Heatmap (first 20 features)
corr = X.iloc[:, :20].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation of First 20 Features (eos4u6p)')
plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/hERG_feature_corr.png')
plt.close()

# PCA Scatter (by label)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='skyblue', label='0 (Non-blockers)', alpha=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='salmon', label='1 (Blockers)', alpha=0.5)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('hERG PCA (eos4u6p Features)')
plt.legend()
plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/hERG_pca_labels.png')
plt.close()
