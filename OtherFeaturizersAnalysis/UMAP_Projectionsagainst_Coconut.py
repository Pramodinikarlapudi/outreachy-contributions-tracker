import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the already calculated Coconut features
data = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_coconut_features.csv")
y = data['Label']

# Already calculated UMAP1 and UMAP2
X_umap = data[['UMAP1', 'UMAP2']].values

# Plot UMAP
plt.scatter(X_umap[y == 0, 0], X_umap[y == 0, 1], c='skyblue', label='0 (Non-blockers)', alpha=0.5)
plt.scatter(X_umap[y == 1, 0], X_umap[y == 1, 1], c='salmon', label='1 (Blockers)', alpha=0.5)
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('hERG UMAP (eos8ub5 Features)')
plt.legend()
plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/Coconut_hERG_umap_labels.png')
plt.close()