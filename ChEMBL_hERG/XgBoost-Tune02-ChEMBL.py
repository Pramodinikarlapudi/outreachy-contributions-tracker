import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


xgb = XGBClassifier()
xgb.load_model("/mnt/d/outreachy-contributions-tracker/models/hERG_xgb_tune02_model.json")  # Adjust if needed

chembl_data = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_Chembl_Morgan_fingerprints_features.csv")


chembl_original = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_chembl_50.csv")  # Adjust path!
print("chembl_original columns:", chembl_original.columns.tolist())
print("chembl_original sample:\n", chembl_original.head())


chembl_data = chembl_data.merge(chembl_original[['Smiles', 'LABEL (Y)']], 
                                left_on='SMILES', 
                                right_on='Smiles', 
                                how='left')


print("Missing labels in chembl_data['LABEL (Y)']:", chembl_data['LABEL (Y)'].isna().sum())
print("chembl_data sample after merge:\n", chembl_data[['SMILES', 'LABEL (Y)']].head())


chembl_data = chembl_data.dropna(subset=['LABEL (Y)'])
signature_columns = [col for col in chembl_data.columns if col not in ['SMILES', 'Label', 'Smiles', 'LABEL (Y)']]

X_test = chembl_data[signature_columns].values
y_test = chembl_data['LABEL (Y)'].astype(int).values

y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy on 50 ChEMBL compounds: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# ROC - AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {np.trapezoid(tpr, fpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - hERG XGBoost (ChEMBL Test)')
plt.legend()
plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/hERG_ChEMBL_xgb_tune2_roc_curve.png')
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - hERG XGBoost (ChEMBL Test)')
plt.savefig('/mnt/d/outreachy-contributions-tracker/figures/hERG_ChEMBL_xgb_tune2_confusion_matrix.png')
plt.close()

chembl_data['Predicted_Label'] = y_pred
chembl_data.to_csv("/mnt/d/outreachy-contributions-tracker/ChEMBL_hERG/hERG_chembl_50_predictions_xgb_tune2.csv", index=False)
print("Predictions saved to hERG_chembl_50_predictions_xgb.csv")