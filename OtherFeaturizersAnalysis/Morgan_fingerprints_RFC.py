import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Featurized data loading

data = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_Morgan_fingerprints_features.csv")
X = data.drop(columns=['SMILES', 'Label'])
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

rf = RandomForestClassifier(n_estimators = 100, random_state = 42, class_weight = 'balanced')
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {np.trapezoid(tpr, fpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - hERG RFC')
plt.legend()
plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/hERG_Morgan_fingerprints_roc_curve.png')
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - hERG RFC')
plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/hERG_Morgan_fingerprints_confusion_matrix.png')
plt.close()

