import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_Morgan_fingerprints_features.csv")
X = data.drop(columns=['SMILES', 'Label'])
y = data['Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, gamma=0.5, min_child_weight=5, random_state=42, scale_pos_weight=1.5)
xgb.fit(X_train, y_train)

xgb.save_model("/mnt/d/outreachy-contributions-tracker/models/hERG_xgb_tune02_model.json")
print("Model saved to hERG_xgb_tune02_model.json")


y_prob = xgb.predict_proba(X_test)[:, 1]


threshold = 0.55
y_pred_thresh = (y_prob >= threshold).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred_thresh)
precision = precision_score(y_test, y_pred_thresh)
recall = recall_score(y_test, y_pred_thresh)

print(f"\nUsing threshold = {threshold}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# ROC - AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {np.trapezoid(tpr, fpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - MFP XGBoost Tuned')
plt.legend()
plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/hERG_MFP_tune_xgb_roc_curve.png')
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_thresh)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix_MFP_Threshold {threshold}')
plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/hERG_MFP_tune_xgb_confusion_matrix_thresh.png')
plt.close()
