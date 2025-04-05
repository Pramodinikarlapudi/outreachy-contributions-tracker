## Downloading hERG and Tox21 datasets from TDC (Therapeutics Data Commons), saving them to Google Drive and Git repo.  

Tools we'll use -

1. Google Colab
2. Google Drive
3. Git Bash

**Setting up Colab :**

Open a new notebook in Google Colab.  

Install the TDC library using  
```
!pip install PyTDC
```

(It'll take a few seconds, we'll see "Successfully installed PyTDC" when it is done).  

I picked the hERG (Karim et al.) dataset from TDC. It has 13,445 compounds, but **scaffold split** gave me 655 (I feel it is decent).

To download hERG Dataset (Run this code):

```
from tdc.single_pred import Tox
data = Tox(name='hERG')
```

```
split = data.get_split(method='scaffold')
```

For hERG, **it took 13,445 compounds** and gave me a **smaller set** i.e., 458 train, 66 valid, 131 test (655 total).

(**Next step is heavily customizable -- you can choose not to save your datasets in your drive**).

If you decide to save the dataset files in your drive (like me), first we'll have to connect Colab to your Drive.

```
from google.colab import drive
drive.mount('/content/drive')
```

On running this code, you will get directed to **sign-in page** of your Google Drive (Sign-in, allow permissions and let's proceed).

After you come back to your Colab, It should say **"Mounted at /content/drive"** .

Now, let's save the hERG split to a folder in your Drive.  

```
split['train'].to_csv('/content/drive/MyDrive/TDC-Datasets/hERG_train.csv', index=False)
split['valid'].to_csv('/content/drive/MyDrive/TDC-Datasets/hERG_valid.csv', index=False)
split['test'].to_csv('/content/drive/MyDrive/TDC-Datasets/hERG_test.csv', index=False)

```

This makes a '**TDC-Datasets**' folder in 'MyDrive' with the three files (Train, Valid, Test).

We're done with installing our libraries and saving our files in GoogleDrive.

Next step is to download the files into our machine and push them into our repositories.

To download the files from our drive (Go to the drive > TDC-Datasets folder > right click on files and click on download).

Open your Git Bash terminal and push the data set files to your repository.  

Please refer to [This Video](https://youtu.be/YV74aapk72A?si=oWJm35VK9-bY9f7m), if you aren't able to follow these steps / face any issues.  

**3D Molecule Visualizations**  // Optional 

I have converted a very small subset of SMILES strings from the hERG dataset into 3D SDF files.  

The first three SMILES strings from the 'Drug' Column of 'hERG_train.csv', 'hERG_valid.csv', 'hERG_test.csv' (Combined)  

**SMILES List**

1. `Oc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1`
   
   ![hERG_mol_1](https://github.com/user-attachments/assets/06970794-5ff5-4737-8ee0-02db874efa49)

2. `Fc1ccc(C(OCC[NH+]2CC[NH+](CCCc3ccccc3)CC2)c2ccc(F)cc2)cc1`
   
   ![hERG_mol_2](https://github.com/user-attachments/assets/5e110e00-ffac-4764-b286-8225348565df)

3. `CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.O=P([O-])([O-])[O-]`
   
   ![hERG_mol_3](https://github.com/user-attachments/assets/5098da8e-e8ac-4470-9e1b-316f80157312)

**Step01**

SMILES Extraction (from CSV files to create a manageable subset)  

refer **notebooks/extract_smiles.py**  

```
import pandas as pd

train = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_train.csv")
valid = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_valid.csv")
test = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_test.csv")

```

```
all_smiles = pd.concat([train['Drug'], valid['Drug'], test['Drug']]).head(3)
all_smiles.to_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_3_smiles.txt", index=False, header=False)

```

**Step 02**  

You'll need Open Babel (to help you convert SMILES strings to a seperate 3D SDF file manually)  

(I have done it manually to ensure accuracy but, can just be done all at a time)  

```
echo "Oc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1" | obabel -i smi -o sdf -O data/hERG_mol_1.sdf --gen3D
echo "Fc1ccc(C(OCC[NH+]2CC[NH+](CCCc3ccccc3)CC2)c2ccc(F)cc2)cc1" | obabel -i smi -o sdf -O data/hERG_mol_2.sdf --gen3D
echo "CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.O=P([O-])([O-])[O-]" | obabel -i smi -o sdf -O data/hERG_mol_3.sdf --gen3D

 ```

Above commands create individual SDF files, each with 3D coordinates generated using **--gen3D**.  

**Step 03**

We'll be seeing only some random numbers in our saved SDF files.  

Download **Avogadro** from avogrado.com for being able to visualize the 3D structures using our saved files.  

Open each SDF file to generate 3D visualizations with the "Ball and Stick" display style.  

```

avogadro /mnt/d/outreachy-contributions-tracker/data/hERG_mol_1.sdf

```

We can repeat the same command for all the molecules we want to visualize (understand it's structure).  

This command automatically opens avogadro software and you will be directed to the visualization.  

path  
- **notebooks/visuals_hERG.py**
- **figures/hERG_label_dist.png**  
     
**Visualizing raw hERG dataset**  
![hERG_label_dist](https://github.com/user-attachments/assets/7353466f-0b2d-4e66-ab15-8e055afb8309)  

Our data is slightly imbalanced (it has 70:30 ratio in every set), let's featurize our data now.   

## Outlining featurization process of hERG dataset using Ersilia representation model.  

I selected '**eos4u6p**' (**Chemical checker signaturizer**) and implemented it in a python script. 

Explored [Ersilia Model Hub](https://ersilia.io/model-hub) for "**Representation**" models that convert SMILES into numerical features suitable for machine learning.  

Reviewed models like '**eos8ub5**' (Coconut projections, 8 dims), '**eos2db3**' (DrugBank projections, 8 dims), '**eos8f2t**' (Scaled WHALES, 11 dims), and '**eos4u6p**' (Chemical checker signaturizer, 3200 dims).  

**Model Chosen**  '**eos4u6p**' - "Chemical Checker 25 Bioactivity Signatures (2020-02 version)"  

It generates 25 signatures (128 dimensions each) totaling **3200 features per compound**, capturing 2D/3D fingerprints, scaffolds, binding affinities, side effects and cell bioassay data.  

hERG cardiotoxicity depends on ion channel binding and side effects -- **eos4u6p**'s signatures directly target these, unlike pure structural models (Eg. Coconut projections)  

3200 dimensions provide a detailed view making it ideal if i apply Random Forest.  (Helps my model uncover complex patterns in hERG blockage).  

'**eos4u6p**' has been trained on diverse bioactivity data making it suitable for hERG's synthetic drug-like compounds over natural product focused models.  

But, it was **computionally challenging** -- 41MB output. WSL Handled it (Took complete 45 mins though :( )  

**Implementing Featurization**  

- I created a 'featurize.py' in 'notebooks/' folder.  

- Combined 'hERG_train.csv', 'hERG_valid.csv' & 'hERG_test.csv' into 655 SMILES using **pandas**  

- Used 'ErsiliaModel("eos4u6p") to fetch and serve model.  

- Processed SMILES in batches of 20 to manage memory (as 3200 dims is huge).  

- Saved features to 'data/hERG_ccsign_features.csv' - 655 rows, 3202 columns (3200 features + SMILES + Label).  

Please refer to '**notebooks/featurize.py**' for full implementation.  

**Implemented in WSL Ubuntu with Conda environment 'ersilia'** (Also Python3, RDKit, pandas, numpy installed).  

Activate environment  

```
conda activate ersilia

```

Fetch and serve model  

```
ersilia -v fetch eos4u6p
ersilia serve eos4u6p

```

Finally run featurization.  

```
cd /mnt/d/outreachy-contributions-tracker/notebooks $ python3 featurize.py

```

Initial NoneType error was handled by the model's "conventional run" - **no data cleaning needed**  

**hERG_ccsign_features.csv** contains 655 compounds, **each with 3200 bioactivity features**.  

**Evaluating Featurized data 'eos4u6p'**  

path  
- **notebooks/visuals_hERG.py**  
- **figures/hERG_feature_corr.png**  

1. **Correlation HeatMap**

   ![hERG_feature_corr (2)](https://github.com/user-attachments/assets/9c7e5197-4700-4675-bdd8-0acb883d9dff)

   Heatmap helps us evaluate our featurized data in detail.

   - **Diagonal is dark red** indicating every feature is perfectly correlated with itself (**this is observed in every correlation heatmap**)
   - **Red (closer to +1.0)** indicates **strong positive correlation** among features (when one feature increases, the other also increases)
   - **Blue (closer to -1.0)** indicates **strong negative correlation** among features (i.e., inversely proportional. When one feature increases, the other decreases)
   - **White/Gray** (approx / near 0.0) indicates **little to no correlation** (i.e., features are independent)

In this heatmap, **most non-diagonal values are light colored**, meaning weak or no correlation among most features.  
Some features have mild positive correlation (**light red patches**), some have mild negative correlation (**light blue patches**).

Overall, **there is no extreme correlations** apart from the diagonal, suggesting that **hERG dataset doesn't have highly redundant features**.  

This is a **very good sign** as **highly correlated features lead to multicollinearity**, which can affect our model's performance.  

path  
- **notebooks/visuals_hERG.py**  
- **figures/hERG_pca_labels.png**  

2. **PCA Analysis 'eos4u6p'**

   ![hERG_pca_labels](https://github.com/user-attachments/assets/ed3349fa-ea9c-45d8-ba47-75cdfeb592de)

   Points are spread across **PCA1 (-2 to 3)** and **PCA (-2 to 2)**, showing how the compounds differ in their bioactivity signatures.

   - There is **significant** overlap between blockers and non-blockers, especially around PCA1 = 0 to 1 and PCA2 = -1 to 1.
   - This indicates many compounds have **similar bioactivity profiles** despite different hERG outcomes.
   - **Non-blockers** tend to cluster more **on the left** (PCA1 < 0), suggesting a bioactivity pattern distinct from blockers.
   - **Blockers** tend to cluster more **on the right** (PCA1 > 1), with a noticeable cluster around PCA1 = 2 to 3, PCA2 = 1 to 2 indicating some blockers have a unique bioactivity signature.
   - A few **outliers** (PCA ~= 3, PCA ~= 2) are blockers, showing **extreme bioactivity differences**

 The **Significant overlap** suggests that **bioactivity signatures alone don't fully seperate blockers from non-blockers**.

 We can featurize our dataset using different models using same procedure.

 I have experimented with models 
    - **Projections against Coconut** - 'eos8ub5'
    - **Chemical space 2D projections against ChemDiv** - 'eos2db3'
    - **Chemical space 2D projections against DrugBank** - 'eos9gg2'

Let's evaluate **featurized data** using other models as well.  

path  
- **OtherFeaturizersAnalysis/Visuals02.py**  
- **figures/Coconut_hERG_pca_labels.png**  
- **OtherFeaturizersAnalysis/UMAP_Projectionsagainst_Coconut.py**  
- **figures/Coconut_hERG_umap_labels.png**  

1. **PCA plot** of **Projections against Coconut** 'eos8ub5'

   ![Coconut_hERG_pca_labels (1)](https://github.com/user-attachments/assets/cbeec57c-13b0-4d95-923f-6be42a2ff788)

   **UMAP plot** of **Projections against Coconut** 'eos8ub5'

   ![Coconut_hERG_umap_labels (1)](https://github.com/user-attachments/assets/54e29663-5311-49f1-ba88-529dffcb687d)


   **HEAVY Overlap** very difficult to classify using this featurized data.

2. **PCA plot** of **Chemical space 2D projections against ChemDiv** 'eos2db3'

   ![chemDiv_hERG_pca_labels (1)](https://github.com/user-attachments/assets/22537067-7b3d-4144-ac7e-34bb44879baa)

   **UMAP plot** of **Chemical space 2D projections against ChemDiv** 'eos2db3'

   ![chemDiv_hERG_umap_labels (1)](https://github.com/user-attachments/assets/235a52ff-754d-4d0e-a49c-710e43a6ed22)

3. **PCA plot** of **Chemical space of 2D projections against DrugBank** 'eos9gg2'

   ![DrugBank_hERG_pca_labels (1)](https://github.com/user-attachments/assets/3e1e750a-6c6b-42dd-b7ed-fe2e591de7d6)

   **UMAP plot** of **Chemical space 2D projections against DrugBank** 'eos9gg2'

   ![DrugBank_hERG_umap_labels (1)](https://github.com/user-attachments/assets/01ff4292-28b2-4807-a299-fcf2aff111be)

On comparing all four model's featurized data, we can clearly understand that our first choice of featurizer **Chemical checker signaturizer** - **'eos4u6p'** gives us more classifiable data.  

Now, **let's train our model using non-linear models** like **Random Forest**, **XG-Boost** or **Support Vector Machine (SVM)** as PCA visualization showed overlap between blockers and non-blockers, **suggesting non-linear relationships in the bioactivity space**.

## Model Training  

Let's now train our hERG dataset using a '**Random Forest Classifier (RFC)**', it has a strong baseline performance on binary classification tasks.  

For our **hERG** classification, dataset is imbalanced and **RFC** provides features like 

```
class_weight='balanced'
```

Also, **RFC** makes no assumptions about feature distributions, relationships between features and labels which is **Ideal** for us.


for detailed implementation refer - **notebooks/trainRFC_hERG.py**  

**Install Necessary libraries** - Obvious first step :)

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```
from sklearn.ensemble import RandomForestClassifier
from skelarn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, confusion_matrix
```

**Train the model using featurized data**

```
data = pd.read_csv("/mnt/d/outreachy-contributions-tracker/data/hERG_ccsign_features.csv")
X = data.drop(columns=['SMILES', 'Label'])
y = data['Label']
```
Customize the path as needed.

**Splitted data into 80-20 train-test sets (524 train, 131 test)**  Can use different ratio but make sure you are giving your model enough data to train on.  

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
**random_state=42** setting makes sure we are getting same results on each run.  

**Key Hyperparameter settings**

1. **n_estimators=100**
2. **class_weight='balanced'**
3. **random_state=42**

**n_estimators** - No.of.Trees we would like to train our data on, I chose 100 (Neither too less nor too many trees). More number of trees leads to overfitting (Our model learns unnecessary noise in the training data)  

**class_weight='balanced'** - As visualized, our dataset is imbalanced, hence for giving equal importance to both the classes (blockers and non-blockers) we used balanced feature.  

```
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
```

**Evaluating our model (RFC)**

| Accuracy | Precision | Recall |
|----------|-----------|--------|
| 0.79 | 0.80 | 0.95 |

We are evaluating RFC model's performance on **20%** of test set (**131 compounds**)

**Accuracy - 0.79** - Indicating 79% of predictions were correct.  
**Precision - 0.80** - From the compounds predicted as blockers, only 80% were actual blockers.   
**Recall - 0.95** - Model Identified 95% of actual blockers in the test set.   

**High Recall indicates that model is excellent at detecting hERG blockers and non-blockers, reducing the risk of missing dangerous compounds -- Crucial**

**ROC-AUC Curve**  

![hERG_roc_curve](https://github.com/user-attachments/assets/0a72cc17-5177-4beb-ae5a-a38bf73239a6)

**AUC = 0.86** indicates that our RFC has an **86%** probablility of ranking a randomly chosen positive instances **higher** than a randomly chosen negative ones.  

**Our RFC is effectively identifying hERG blockers**


**Confusion Matrix**  

![hERG_confusion_matrix](https://github.com/user-attachments/assets/0d76b1df-b2ed-4890-90d0-34a39bd589e7)

| Actual vs Predicted | Predicted: 0| Predicted: 1 |
|---------------------|-------------|--------------|
| Actual: 0 | 16 | 22 |
| Actual: 1 | 5 | 88 |

1.**True Positives (TP)** - Correctly classified 88 samples as positive. (Model did very well in identifying hERG blockers).  
2.**True Negatives (TN)** - Correctly classified 16 samples as negative.  
3.**False Positives (FP)** - Model **incorrectly classified 22 negative samples as positive** (**Very high false alarm rate**).  
4.**False Negatives (FN)** - Model missed 5 actual positive cases (Very good job here)  

We need to improve our **False Positive Rate** & also try improving **Recall**

**RFC Tuning**

**Key hyperparamater changes**

1. **n_estimators=200** (Increased no.of.trees from **100** to **200**)
2. **max_depth=10** (Limited tree's depth to 10 hoping to **lessen overfitting**)
3. **class_weight={0:0.45, 1:0.55}** (Changed from balanced, assigned weights to each class, model will pay **slightly more attention to class1**)

Path - **notebooks/tuneRFC_hERG.py**  

```
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight={0:0.45, 1:0.55}, random_state=42)
```
```
rf.fit(X_train, y_train)
```

With these changes, i have expected to increase **accuracy** & **Precision**.  

But, results are not good.  

| Accuracy | Precision | Recall |
|----------|-----------|--------|
| 0.79 | 0.79 | 0.95 |  

**Accuracy** stayed same & **Precision** decreased by **0.01**, **Recall** remained same.  

**The reason could be limiting depth of the trees!!**  

**ROC-AUC** Curve - Tuned RFC.  

![hERG_tuned_roc_curve (1)](https://github.com/user-attachments/assets/0e07d588-58b2-4337-a9c1-a245e613f7d9)  

**AUC** moved down from **0.86** to **0.84**  

**Confusion Matrix - Tuned RFC**  

![hERG_tuned_confusion_matrix (1)](https://github.com/user-attachments/assets/b95d76d6-baf5-41f7-bbfe-e97232807cc4)  

| Actual vs Predicted | Predicted: 0 | Predicted: 1|
|---------------------|--------------|-------------|
|Actual: 0 | 15 | 23 |
|Actual: 1 | 5 | 88 |

False positives **Increased** instead of **Decreasing**  

**Not a good tune**  

We can fine-tune RFC this time prioritizing the other class. But, let's try Xg-Boost.  

**Xg-Boost**  

- Unlike RFC, Xg-Boost builds decision trees sequentially. Each new tree learns from the errors of the previous tree. (This can help increase accuracy which is to be improved as it was just 0.79 when trained with RFC).
- In Xg-Boost, trees that perform poorly are given more weight in subsequent iterations. (Also can help increase our accuracy).

**Key Hyperparamater settings**  

1. **n_estimators=100** (Kept tree count optimal as more could lead to overfitting)
2. **learning_rate=0.1** (I use 0.1 in general as it is neither too low like 0.01 nor too high like 0.2)
3. **max_depth=5** (Depth of 5-7 is considered optimal, if lower it would better generalize but 5 is not large so...)
4. **scale_pos_weight= 0.82** (Considered number < 1 as this helps majority class -- we have more 1's (blockers) prioritizing their prediction. But, **will consider weight > 1 while tuning as this can help with dealing 0's** which is our problem now)

```
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, scale_pos_weight=0.82)
xgb.fit(X_train, y_train)
```
Implemented in **models/train_xgb_hERG.py**

My model predicts **1's** very good (**95 out of 100**) but lacks performance while detecting **0's** (non-blockers) -- It predicts non-blockers as blockers :(

Post analyzing the results, i would like to experiment with **learning_rate** & **scale_pos_weight**  

**Evaluation**  

| Accuracy | Precision | Recall |
|----------|-----------|--------|
|0.81 | 0.83 | 0.92 |  

**Accuracy**  

Moved from **0.79** to **0.81** 🥇  

**Precision**

Moved from **0.80** to **0.83** 🥇

**Recall**

Moved from **0.95** to **0.92** (Need a lil tuning) 👍

- **RFC's recall was 0.95** (Correctly classified 95% of actual positive cases).

- **Xgb's recall is 0.92** (Correctly classified 92% of actual positive cases).

**ROC-AUC**  
![hERG_xgb_roc_curve](https://github.com/user-attachments/assets/4372f89d-3e23-4942-b623-6ff63bd406ed)

- **AUC moved from 0.86 to 0.85**

**Confusion Matrix**  

![hERG_xgb_confusion_matrix](https://github.com/user-attachments/assets/779e2d9c-e815-4378-b2ee-2ed2f9276662)  

| Actual vs Predicted | Predicted: 0 | Predicted: 1|
|---------------------|--------------|-------------|
|Actual: 0 | 20 | 18 |
|Actual: 1 | 7 | 86 |

**False positives reduced from 22 False positives to 18** 🥇

**True Negatives increased from 16 true negatives to 20** 🥇

**False Negatives Increased from 5 to 7** 👎

Though Accuracy & Precision improved in Xgb, we cannot prioritize it over RFC's results as False Negatives play a key role in blockage detection.

**We can't be risking 2 more blockers being classified as non-blockers** for just numbers in our evaluation metrics.

Next Goal - To **decrease false positives** protecting both accuracy and precision, Will tune **learning_rate** & **scale_pos_weight**.  


**Xg-Boost Tuned**  

**Key hyperparameters changes**

- **n_estimators=200** (Increased trees from 100 to 200 with a hope that model learns more bioactivity differences)
- **learning_rate=0.05** (Slowed down learning rate significantly, so model could learn better)
- **max_depth=6** (Slightly increased the depth from 5 to 6)
- **scale_pos_weight=1.5** (Favouring minority class **non-blocker's** to minimize false positives)

```
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, scale_pos_weight=1.5)
xgb.fit(X_train, y_train)
```
  
Implemented in **models/tune_xgb_hERG.py**  

**Evaluation**  

| Accuracy | Precision | Recall |
|----------|-----------|--------|
| 0.82 | 0.83 | 0.94 | 

**ROC-AUC Curve**  

![hERG_tune_xgb_roc_curve](https://github.com/user-attachments/assets/932df412-8f54-442f-bb16-4992d1c2fb86)  

**AUC moved from 0.84 (Xgb) to 0.85 (tuned Xgb)**  

**Confusion Matrix**  

![hERG_tune_xgb_confusion_matrix](https://github.com/user-attachments/assets/071f034c-ecdd-487f-bf87-282729c54dcd)  

| Actual vs Predicted | Predicted: 0| Predicted: 1|
|---------------------|-------------|-------------|
| Actual: 0 | 20 | 18 |
| Actual: 1 | 6 | 87 |

**False Positives decreased slightly (from 7 to 6) 🥇**

**Overall Analysis of models till now**  

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| RFC | 0.79 | 0.80 | 0.95 | 0.86 |
| RFC-Tuned | 0.79 | 0.79 | 0.95 | 0.84 |
| Xg-Boost | 0.81 | 0.83 | 0.92 | 0.85 |
| Xg-Boost Tuned | 0.82 | 0.83 | 0.94 | 0.85 |  

I would consider **XG-Boost Tuned is the best** among all four models.

**Highest accuracy** - More overall correct predictions.
**Highest precision** - Fewer false postives.
**Strong recall** - Slightly less than the RFC but still a good recall.
**ROC-AUC** - Slightly less AUC than RFC.
**XG-Boost tuned has overall good performance**.

**Next best** would be **RFC** as it's **Recall and AUC are the highest**.

Leaderboard's **highest AUROC** is **0.88 ± 0.002**, they state that **RFC / SVM**s when paired with **extended-connectivity fingerprints** consistently outperformed all other recently developed models.

So, next would be learning what '**extended-connectivity fingerprints**' are and then applying RFC.  


















