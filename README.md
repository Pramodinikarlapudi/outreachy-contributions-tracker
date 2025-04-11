# hERG Blockage Prediction 🫀
 - **Dataset details**
 - **Visualizing chemical compounds using [Avogadro software](https://avogadro.cc/)**
 - **Featurizing using `Representation Models` from [Ersilia Model Hub](https://www.ersilia.io/model-hub)**
 - **Training on `RFC` & `Xg-Boost`**
 - **Evaluating Model's performance on `ChEMBL hERG Dataset`**
 - **Implementing research paper [hERGAT](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-00957-x)**

## 📝 Introduction  
The `human ether-a-go-go-related gene` (**hERG**) codes for a protein known as `Kv11.1`, which is the alpha subunit of a potassium ion channel critical for **cardiac electrical signaling**.  
- Scientists have been using `vitro` tests such as `radioligand binding` and `QTc analyses` to assess the inhibitory effects of compounds on the hERG channel.
- [Vitro](https://www.fda.gov/medical-devices/products-and-medical-procedures/in-vitro-diagnostics) & [QTc](https://pubmed.ncbi.nlm.nih.gov/36772881/) tests are both **time consuming** and **financially inefficient** as we have to test every single compound.
- With the increasing **availability of data** related to hERG channel, **deep learning** models have recently emerged as a **more effective** approach to predict hERG blockers.

## 🔎 Dataset details  

I have downloaded hERG dataset from **`TDC (Therapeutic Data Commons)`**, which has **648** drugs.  

**Therapeutic Data Commons (TDC)** brings together high-quality datasets and tasks designed to `support machine learning` in therapeutic science.  

**It's a go to resource for drug discovery ML Workflows**.  

**`Binary Classification`** : Given a **`SMILES`** string, predict whether it blocks(1) or not blocks(0).  

- A compound is considered a hERG blocker, if the `IC50` value is **less than 10µM** and a hERG non-blocker if it is **greater than 10µM**.
- **IC50** represents the concentration of a drug or inhibtor required to `reduce a specific biological process by half`.

**Downloading dataset 📥**  

You can download dataset both from your **`google-colab`** and directly from **`VS Code WSH Terminal`**  

1. **Install the TDC library**

   ```
   !pip install PyTDC
   ```
2. **Download dataset**

   ```
   from tdc.single_pred import Tox
   data = Tox(name='hERG')
   ```
3. **Split the data & Save it in your Drive**

   ```
   split = data.get_split(method='Scaffold')
   ```
   - **Scaffold split method groups compounds by their scaffolds (Internal Structure)**.
   - This would help `distribute` compounds with `different scaffolds across train, valid and test sets`.

   If you are working on **google-colab**, mount your **g-drive**.

   ```
   from google.colab
   drive.mount('/content/drive')
   ```
   - It is always a **`good pratice`** to save your **`dataset in your drive`**.  
   - Post Mounting your drive to your colab, **save a copy of dataset files in your machine on your drive**.
     ```
     split['train'].to_csv('/content/drive/MyDrive/TDC-Datasets/hERG_train.csv', index=False)
     split['valid'].to_csv('/content/drive/MyDrive/TDC-Datasets/hERG_valid.csv', index=False)
     split['test'].to_csv('/content/drive/MyDrive/TDC-Datasets/hERG_test.csv', index=False)
     ```
     This makes **`TDC-Datasets`** folder in '**Drive**' with three files (**Train**, **Test** & **Valid**).

   - Please refer this [**video**](https://youtu.be/YV74aapk72A?si=NvvMEoY1IMJ9XViA), where i explained Installation process in detail.

  **Label Distribution** - **[notebooks/visuals_hERG.py](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/notebooks/visuals_hERG.py)** **[figures/hERG_label_dist.png](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/figures/hERG_label_dist.png)**  
     
 **Visualizing raw hERG dataset**⚡    
![hERG_label_dist](https://github.com/user-attachments/assets/7353466f-0b2d-4e66-ab15-8e055afb8309)  

   **Dataset is slightly imbalanced** (More blockers will help our model prioritize not missing blockers, which is actually a good thing).  

   We can balance it with **hyper parameter tuning** when training models. Let's not complicate things ☕︎.  

## 𓂃🖌 Visualizing compounds in 3D  

Few colorful **3D visualizations** of compounds in our hERG dataset hurts no one ⋆˚✿˖°  

I have **`converted`** a small subset of **`SMILES strings`** from the dataset **`into 3D SDF files`**.

1. **SMILES Extraction** (From CSV files)  - Refer [**notebooks/extract_SMILES.py**](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/notebooks/extract_SMILES.py)

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
2. Install [**Open Babel**](https://openbabel.org/) to `convert SMILES strings into seperate 3D SDF files`.
    I have done it manually to ensure accuracy but, can be done all at a time.
   ```
   echo "Oc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1" | obabel -i smi -o sdf -O data/hERG_mol_1.sdf --gen3D
   echo "Fc1ccc(C(OCC[NH+]2CC[NH+](CCCc3ccccc3)CC2)c2ccc(F)cc2)cc1" | obabel -i smi -o sdf -O data/hERG_mol_2.sdf --gen3D
   echo "CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.O=P([O-])([O-])[O-]" | obabel -i smi -o sdf -O data/hERG_mol_3.sdf --gen3D
   ```
   These commands **create individual SDF files**, each `with 3D Co-ordinates` generated using -gen3D.

3. **Download [Avogadro](https://avogadro.cc/) software**.

   Saved `3D files` feel like they are just random numbers but, `are 3D co-ordinates representing compounds`.

   We'll need a visualization software like [Avogadro](https://avogadro.cc/), to actually see the compounds in 3D space.

   - **Open each SDF file to generate 3D visualizations with the "Ball and Stick" display style**.

     ```
     avogadro /mnt/d/outreachy-contributions-tracker/data/hERG_mol_1.sdf
     ```
   - **Repeat** the same command for all the **molecules you want to visualize**.

 Here's **how first three molecules** from dataset looks like ⋆⭒˚.⋆🪐 ⋆⭒˚.⋆  

 **SMILES** - **Oc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1**  
 
  ![hERG_mol_1](https://github.com/user-attachments/assets/06970794-5ff5-4737-8ee0-02db874efa49)

 **SMILES** - **Fc1ccc(C(OCC[NH+]2CC[NH+](CCCc3ccccc3)CC2)c2ccc(F)cc2)cc1**  
 
   ![hERG_mol_2](https://github.com/user-attachments/assets/5e110e00-ffac-4764-b286-8225348565df)

 **SMILES** - **CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.CCCCCCC[N+](CC)(CC)CCCCc1ccc(Cl)cc1.O=P([O-])([O-])[O-]**  
 
   ![hERG_mol_3](https://github.com/user-attachments/assets/5098da8e-e8ac-4470-9e1b-316f80157312)


## 👩🏻‍💻 Featurizing using Ersilia Representation Models  

**[Ersilia](https://www.ersilia.io/)** offers a growing hub of `pre-trained ML models` designed to support **biomedical** and **chemoinformatics** applications with `minimal setup`.  

There are a **few pre-requisite softwares** needed to support Ersilia on your machine. This [**guide doc**](https://ersilia.gitbook.io/ersilia-book/ersilia-model-hub/installation) will assist you with the installation procedure in detail.  

**Pre-requisite Softwares**  
 - `Python` & `Miniconda`
 - `GCC Compiler`
 - `Git` & `GitHub` CLI
 - `GIT LFS`
 - `Docker`

**System Requirements** 
 - If you are using `Windows` -- Atleast Windows 10 is compulsory.
 - If you are using `Mac OS` -- **Ersilia** can be installed directly. 

**My System Details**  
 - Ubuntu : `24.04`
 - Windows11 : `WSL Integration`
 - Docker : `28.0.1` (build - 06801e)

**Detailed Installation procedure**
 - Create your **conda** environment.
   
   ```
   conda create -n ersilia python=3.12
   ```
 - Activate your **conda** environment.
   
   ```
   conda activate ersilia
   ```
 - Clone the **Ersilia** github repo from your **CLI**.
   
   ```
   git clone https://github.com/ersilia-os/ersilia.git
   ```
 - Change your current directory to **ersilia**.
   
   ```
   cd ersilia
   ```
 - Lastly, Install **Ersilia** in `developer mode`.
   
   ```
   pip install -e .
   ```

**Check if `Ersilia` is properly Installed !!**  

 ```
 ersilia --help
 ```
Your terminal should display something similar to [**this image**](https://private-user-images.githubusercontent.com/97833644/424876158-3b06ad5d-459a-4857-b738-de4658d4d1fc.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQzNTU3NzksIm5iZiI6MTc0NDM1NTQ3OSwicGF0aCI6Ii85NzgzMzY0NC80MjQ4NzYxNTgtM2IwNmFkNWQtNDU5YS00ODU3LWI3MzgtZGU0NjU4ZDRkMWZjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA0MTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNDExVDA3MTExOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTAyMGNjYTVhYTZkZmZlMmY3ZTEwYWRiOGUzY2VhZTExODcwODEzZTE1NDMzYWQzN2I0MTQ1YzQwMzkwODhmNGEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.MTfTXSFMO1QBJOfY7nJjraGEeXdMv3RtbFGVR4GHWFg)

Post successful installation of **`Ersilia`**, explored **`Representation Models`** on **[Ersilia Model Hub](https://www.ersilia.io/model-hub)**.

**Experimented with in total of 5 Featurizers** - **[notebooks/featurize.py](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/notebooks/featurize.py)**

1. **Chemical checker signaturizer** - **`eos4u6p`**
2. **Morgan fingerprints in binary form (radius 3, 2048 dimensions)** - **`eos4wt0`**
3. **Projections against Coconut** - **`eos8ub5`**
4. **Chemical space 2D projections against ChemDiv** - **`eos2db3`**
5. **Chemical space 2D projections against DrugBank** - **`eos9gg2`**

Chose **Chemical checker signaturizer** - **`eos4u6p`** as it featurizes data into **3200 dimensional bioactivity signatures**, which will help models uncover complex patterns in hERG blockage.

- **`eos4u6p`** generates 25 signatures (**128 dimensions each**) totalling **3200 features per compound**.
- Captures **2D/3D fingerprints**, **Scaffolds**, **Binding affinities**, **Side effects** and **Cell Bioassay data**.
- hERG cardiotoxicity **`depends on ion channel binding`** and **`eos4u6p`**'s signatures directly target these. Unlike pure structural models (Eg. **`Projections against Coconut`**).
- **'eos4u6p'** has been trained on **`diverse bioactivity data`** making it suitable for hERG's synthetic drug like compounds over natural product focused models.
- **It'll be computationally challenging ⌛** (So..patience!!)

**📐Implementing Featurization**  - **notebooks/featurize.py**  

- Activate **conda** environment
  
  ```
  conda activate ersilia
  ```
- Combined **'hERG_train.csv'**, **'hERG_valid.csv'**, **'hERG_test.csv'**
- Fetch & Serve Model **`eos4u6p`**  

  ```
  ersilia -v fetch eos4u6p
  ```
  ```
  ersilia serve eos4u6p
  ```
- **Finally Featurize**

  ```
  cd /mnt/d/outreachy-contributions-tracker/notebooks $ python3 featurize.py
  ```

- **No Data Cleaning Needed** : Initial **`NoneTypeError`** was handled by the model's **conventional run** (Basically, on itself).

Featurized data of **655** compounds, each with **3200 bioactivity features** is stored in **notebooks/hERG_ccsign_features.csv**

On analyzing the **correlation between features** post featurization using **`Heat Map`**

![hERG_feature_corr (3)](https://github.com/user-attachments/assets/ffc8b3ee-6f74-4720-83a6-2edc14088f27)  

Heatmap helps us evaluate our featurized data in detail.

   - **Diagonal is dark red** indicating every feature is perfectly correlated with itself (**this is observed in every correlation heatmap**)
   - **Red (closer to +1.0)** indicates **strong positive correlation** among features (when one feature increases, the other also increases)
   - **Blue (closer to -1.0)** indicates **strong negative correlation** among features (i.e., inversely proportional. When one feature increases, the other decreases)
   - **White/Gray** (approx / near 0.0) indicates **little to no correlation** (i.e., features are independent)

In this heatmap, **most non-diagonal values are light colored**, meaning weak or no correlation among most features.  
Some features have mild positive correlation (**light red patches**), some have mild negative correlation (**light blue patches**).

Overall, **there is no extreme correlations** apart from the diagonal, suggesting that **hERG dataset doesn't have highly redundant features**.  

This is a **very good sign** as **highly correlated features lead to multicollinearity**, which can affect our model's performance.  

**PCA Analysis 'eos4u6p'**

   ![hERG_pca_labels](https://github.com/user-attachments/assets/ed3349fa-ea9c-45d8-ba47-75cdfeb592de)

   Points are spread across **PCA1 (-2 to 3)** and **PCA (-2 to 2)**, showing how the compounds differ in their bioactivity signatures.

   - There is **significant** overlap between blockers and non-blockers, especially around PCA1 = 0 to 1 and PCA2 = -1 to 1.
   - This indicates many compounds have **similar bioactivity profiles** despite different hERG outcomes.
   - **Non-blockers** tend to cluster more **on the left** (PCA1 < 0), suggesting a bioactivity pattern distinct from blockers.
   - **Blockers** tend to cluster more **on the right** (PCA1 > 1), with a noticeable cluster around PCA1 = 2 to 3, PCA2 = 1 to 2 indicating some blockers have a unique bioactivity signature.
   - A few **outliers** (PCA ~= 3, PCA ~= 2) are blockers, showing **extreme bioactivity differences**

 The **Significant overlap** suggests that **bioactivity signatures alone don't fully seperate blockers from non-blockers**.

**After studying multiple papers on hERG blockage predictions, I observed that the featurizer `Morgan fingerprints` (also referred to as `Extended connectivity fingerprints`) is being used extensively to featurize the data**

You can use exact same steps to featurize & visualize using any different model on **`Ersilia Model Hub`**.

Refer - **[OtherFeaturizersAnalysis/eos4wt0_featurize.py](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/OtherFeaturizersAnalysis/eos4wt0_featurize.py)**
      - **[data/hERG_Morgan_fingerprints_features.csv](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/data/hERG_Morgan_fingerprints_features.csv)**

**Also, visualize PCA & UMAPs of other featurizers as well (Helps us visualize the differences more in detail)**  

1. **PCA plot** of **Projections against Coconut** 'eos8ub5'

   ![Coconut_hERG_pca_labels (1)](https://github.com/user-attachments/assets/cbeec57c-13b0-4d95-923f-6be42a2ff788)

   **UMAP plot** of **Projections against Coconut** 'eos8ub5'

   ![Coconut_hERG_umap_labels (1)](https://github.com/user-attachments/assets/54e29663-5311-49f1-ba88-529dffcb687d)
   

3. **PCA plot** of **Chemical space 2D projections against ChemDiv** 'eos2db3'

   ![chemDiv_hERG_pca_labels (1)](https://github.com/user-attachments/assets/22537067-7b3d-4144-ac7e-34bb44879baa)

   **UMAP plot** of **Chemical space 2D projections against ChemDiv** 'eos2db3'

   ![chemDiv_hERG_umap_labels (1)](https://github.com/user-attachments/assets/235a52ff-754d-4d0e-a49c-710e43a6ed22)

4. **PCA plot** of **Chemical space of 2D projections against DrugBank** 'eos9gg2'

   ![DrugBank_hERG_pca_labels (1)](https://github.com/user-attachments/assets/3e1e750a-6c6b-42dd-b7ed-fe2e591de7d6)

   **UMAP plot** of **Chemical space 2D projections against DrugBank** 'eos9gg2'

   ![DrugBank_hERG_umap_labels (1)](https://github.com/user-attachments/assets/01ff4292-28b2-4807-a299-fcf2aff111be)

On comparing all model's featurized data, we can clearly understand that our choice of featurizers **Chemical checker signaturizer** & **Morgan fingerprints** gives us more classifiable data.  

**let's train our model using non-linear models** like **Random Forest**, **XG-Boost** or **Support Vector Machine (SVM)** as PCA visualization showed overlap between blockers and non-blockers, **suggesting non-linear relationships in the bioactivity space**.

## </> Training RFC & Xg-boost models - **notebooks/trainRFC_hERG.py**  

**Random Forest Classifier (RFC)** - Strong baseline performance on **`binary classification tasks`**.  

For **hERG** classification, dataset is slightly imbalanced and **RFC** provides features like 

```
class_weight='balanced'
```

Also, **RFC** makes no assumptions about feature distributions, relationships between features and labels which is **Ideal** for us.


**Install Necessary libraries 📥** - Obvious first step :)

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

**Train the model using featurized data 🪐** -- **`eos4u6p`**

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

**Key Hyperparameter settings ⚙️**

1. **n_estimators=100**
2. **class_weight='balanced'**
3. **random_state=42**

**n_estimators** - No.of.Trees we would like to train our data on, I chose 100 (Neither too less nor too many trees). More number of trees leads to overfitting (Our model learns unnecessary noise in the training data)  

**class_weight='balanced'** - As visualized, our dataset is imbalanced, hence for giving equal importance to both the classes (blockers and non-blockers) we used balanced feature.  

```
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
```

**Evaluating our model (RFC) 📖**

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

**Our RFC is effectively identifying hERG blockers 🏅**

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

**RFC Tuning 📍** - **notebooks/tuneRFC_hERG.py**  

**Key hyperparamater changes ⚙️**

1. **n_estimators=200** (Increased no.of.trees from **100** to **200**)
2. **max_depth=10** (Limited tree's depth to 10 hoping to **lessen overfitting**)
3. **class_weight={0:0.45, 1:0.55}** (Changed from balanced, assigned weights to each class, model will pay **slightly more attention to class1**)

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

**The reason could be limiting depth of the trees!! 📖**  

**ROC-AUC** Curve - Tuned RFC.  

![hERG_tuned_roc_curve (1)](https://github.com/user-attachments/assets/0e07d588-58b2-4337-a9c1-a245e613f7d9)  

**AUC** moved down from **0.86** to **0.84** 👎  

**Confusion Matrix - Tuned RFC**  

![hERG_tuned_confusion_matrix (1)](https://github.com/user-attachments/assets/b95d76d6-baf5-41f7-bbfe-e97232807cc4)  

| Actual vs Predicted | Predicted: 0 | Predicted: 1|
|---------------------|--------------|-------------|
|Actual: 0 | 15 | 23 |
|Actual: 1 | 5 | 88 |

False positives **Increased** instead of **Decreasing**  

**Not a good tune ✖**  

We can fine-tune RFC this time prioritizing the other class. But, let's try **Xg-Boost**. 

**Xg-Boost 📍**  

- Unlike RFC, Xg-Boost builds decision trees sequentially. Each new tree learns from the errors of the previous tree. (This can help increase accuracy which is to be improved as it was just 0.79 when trained with RFC).
- In Xg-Boost, trees that perform poorly are given more weight in subsequent iterations. (Also can help increase our accuracy).

**Key Hyperparamater settings ⚙️**  

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

**Xg-Boost Tuned 📍**  

**Key hyperparameters changes ⚙️**

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

**Overall Analysis of models till now 🛸**  

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

🪐**Next best** would be **RFC** as it's **Recall and AUC are the highest**.

Leaderboard's **highest AUROC** is **0.88 ± 0.002**, they state that **RFC / SVM**s when paired with **extended-connectivity fingerprints** consistently outperformed all other recently developed models.  

So, next would be training on data that is featurized by '**extended-connectivity fingerprints**' 🚀.  

**RFC on `eos4wt0` ☄️** - **[OtherFeaturizersAnalysis/Morgan_fingerprints_RFC.py](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/OtherFeaturizersAnalysis/Morgan_fingerprints_RFC.py)**  

Same **default RFC model** is now trained on **Morgan fingerprints featurized data**  - **[data/hERG_Morgan_fingerprints_features.csv](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/data/hERG_Morgan_fingerprints_features.csv)**  

**`Results`** significantly **Improved**  

**ROC-AUC**  

![hERG_Morgan_fingerprints_roc_curve](https://github.com/user-attachments/assets/ea90c9fc-3dd4-40a2-8f13-c1ec6ee17a3d)  

**Confusion Matrix**  
![hERG_Morgan_fingerprints_confusion_matrix](https://github.com/user-attachments/assets/f8917de9-d7b4-4500-a7a1-ff60384d3bc9)


**Xg-Boost tuned** - **[OtherFeaturizersAnalsysis/Morgan_fingerprints_Xg-Boost_tuned.py](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/OtherFeaturizersAnalysis/Morgan_fingerprints_Xg-Boost_tuned.py)**  

**ROC-AUC**  

![hERG_Morgan_fingerprints_tune_xgb_roc_curve](https://github.com/user-attachments/assets/fff43e36-fc3f-4dad-af6a-a54bf55e68a2)  

**Confusion Matrix**  

![hERG_Morgan_fingerprints_tune_xgb_confusion_matrix](https://github.com/user-attachments/assets/8dc6419f-772b-419f-8d88-9a758a9b01a9)  

**Xg-Boost Tune-02**  - **[Models/fingerprints_xgb_tune02.py](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/models/Morgan_fingerprints_xgb_tune02.py)**

- **Key changes**
```
gamma=0.5 (Controlled unnecessary splits)
min_child_weight=5 (removed weak branches that don't have enough data supporting them)
threshold=0.55 (More cautious before classifying the compound as blocker - as i needed to reduce False Positives).

```
Results were **pretty good** and I was **able to achieve the ROC-AUC of 0.88** (Which is the highest in the TDC leader board).  

**ROC-AUC**  

![hERG_MFP_tune_xgb_roc_curve](https://github.com/user-attachments/assets/d8ad0b91-169a-4581-926e-08c57e1b53c0)  

**Confusion Matrix**  

![hERG_MFP_tune_xgb_confusion_matrix_thresh](https://github.com/user-attachments/assets/a1c93c06-f6e0-4ea4-b88b-5da7f5e2f44c)  

**Clear Summary of Results**  

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| **RFC** | 0.81 | 0.81 | 0.96 | 0.84 |
| **Xg-Boost Tuned** | 0.83 | 0.83 | 0.97 | 0.86 |
| **Xg-Boost Tune02** | **0.83** | **0.84** | **0.95** | **0.88** |

## Evaluating performance ⋆⭒˚.⋆🔭  

| Featurizer | Model | Accuracy | Precision | Recall | ROC-AUC |
|------------|-------|----------|-----------|--------|---------|
| **`eos4u6p`** | **RFC** | 0.79 | 0.80 | 0.95 | 0.86 |
| **`eos4u6p`** | **RFC-Tuned** | 0.79 | 0.79 | 0.95 | 0.84 |
| **`eos4u6p`** | **Xg-Boost** | 0.81 | 0.83 | 0.92 | 0.85 |
| **`eos4u6p`** | **Xg-Boost Tuned** | 0.82 | 0.83 | 0.94 | 0.85 |
| **`eos4wt0`** | **RFC** | 0.81 | 0.81 | 0.96 | 0.84 |
| **`eos4wt0`** | **Xg-Boost Tuned** | **0.83** | 0.83 | **0.97** | 0.86 |
| **`eos4wt0`** | **Xg-Boost Tune02** | **0.83** | **0.84** | 0.95 | **0.88** |

Model's performance **Significantly** Increased on using **eos4wt0**  

## 📍Testing on ChEMBL hERG Dataset  

**`ChEMBL`** is a large-scale bio activity database containing detailed experimental data for `drug-like` compounds, Ideal for `evaluating ML models` in therapeutic research.  

**Downloading hERG dataset in detail 📥**  

- Visit **[ChEMBL](https://www.ebi.ac.uk/chembl/)**, choose your target **`hERG`**
- On selecting the target as **`hERG`** you should be directed to **[detailed filters page](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL240#NameAndClassification)**
- Select **Name and Classification** : **Ion Channel** (check left nav bar).
- On selecting **Ion Channel** -> You'll be directed to **[detailed features filtering page](https://www.ebi.ac.uk/chembl/explore/targets/STATE_ID:aXly8SZHDEs7DcFYkU618A%3D%3D)**
- **Filters** - Organism Taxonomy L1 : **`Eukaryotes`**
              - Organism : **`Homo sapiens`**
              - Protein Classification : **`Ion Channel`**
- **Post applying necessary filters Download the dataset in `CSV format`**. (Check top right 👀)
  
**Dataset** downloaded has around `1,000+` compounds, I have selected the first few compounds and cleaned the data. 

**(Removed duplicate SMILES, Blank IC50 information rows, also removed extra physico-chemical relation information Eg. ALOGP, Molecular Weight...)**  

**Notable Points**

- Made sure **`training set`** and **`ChEMBL dataset`**'s features match.
- We can visualize whether **both dataset's chemical space alignment** using PCA plot.
- **`Orange`** dots indicates **ChEMBL compound's features**.
- **`Blue`** dots indicate **TDC - Training set compound's features**.
- Path : **[figures/PCA_TDC_vs_ChEMBL.png](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/figures/PCA_TDC_vs_ChEMBL.png)**
       : **[ChEMBL_hERG/PCA_features_comparison.py](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/ChEMBL_hERG/PCA_features_comparision.py)**

  ![PCA_TDC_vs_ChEMBL](https://github.com/user-attachments/assets/8fc41c32-de9a-4a48-b6fc-49e4d9d50696)

  **PCA clearly indicates the alignment of chemical space**

**Featurized using `eos4wt0` - `Morgan fingerprints in binary form`** - **[OtherfeaturizersAnalysis/chembl_Morgan_featurize.py](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/OtherFeaturizersAnalysis/chembl_Morgan_featurize.py)**  

Tested on both **XgBoost-tuned** and **XgBoost-Tune02**  

- [ChEMBL_hERG/XgBoost-Tuned-ChEMBL.py](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/ChEMBL_hERG/XgBoost-Tuned-ChEMBL.py)
- [ChEMBL_hERG/XgBoost-Tune02-ChEMBL.py](https://github.com/Pramodinikarlapudi/outreachy-contributions-tracker/blob/main/ChEMBL_hERG/XgBoost-Tune02-ChEMBL.py)

**Results** are as follows :

**Confusion Matrix - XgBoost Tuned**  

![hERG_ChEMBL_xgb_confusion_matrix](https://github.com/user-attachments/assets/287efcc8-1e36-4eac-8280-c92680878e03)  

**Confusion Matrix - XgBoost Tune-02**  

![hERG_ChEMBL_xgb_tune2_confusion_matrix](https://github.com/user-attachments/assets/47e6053b-3133-4b0e-90c2-53a57e9924d7)  

**Model has alarming False Positive Rate** but **It is the best at predicting blockers**  

**ROC-AUC - XgBoost Tuned**  

![hERG_ChEMBL_xgb_roc_curve](https://github.com/user-attachments/assets/1dbe498d-44b2-4927-a8e9-a1344b79da5f)  

**ROC-AUC - XgBoost Tune02**  

![hERG_ChEMBL_xgb_tune2_roc_curve](https://github.com/user-attachments/assets/432e6fba-981e-458d-a607-a436d7780a87)  

**Model is great at predicting blockers but misclassifies non-blockers as blockers**  

## 🔬 Implementing [hERGAT paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-00957-x)  
 
**[hERGAT](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-00957-x)** approach promises significant good results 😌  

**Experimental performance comparison of hERGAT model on test dataset**  

| Method | Accuracy | AUROC | AUPR | F1-score |
|--------|----------|-------|------|----------|
| **hERGAT** | **0.872** | **0.907** | **0.904** | **0.891** |  

Which is so much more efficient compared to both of our **`Xg-Boost`** and **`RFC`** models.  

**This work has been submitted this January and is THE BEST approach with significant reliability**  

**Data Collection** - Collected datasets from **five** different sources.  

1. **ChEMBL** - `7,431` compounds
2. **PubChem** - `5,931` compounds
3. **Li Q**
4. **Wang S**       -- `16,192` compounds from 4 datasets
5. **Zhang Y**
6. **Kim H**

![Figure 1](https://github.com/user-attachments/assets/badae6a9-4afd-427b-87eb-82b02f9b5190)  

**Same conditions** - A compound is considered a `hERG blocker`, if the `IC50 value is less than 10µM` and a `hERG non-blocker` if it is `greater than 10µM`.

- Used **SMILES** to represent each compound's molecular formula through `ASCII` strings.
- Used **RDKit** library to **standardize SMILES**. (Due to possibility of a single compound having multiple valid SMILES representations)

**After Integrating all datasets, Integrated dataset consisted of 23,381 unique compounds. 14,183 hERG Blockers & 9,198 hERG Non-Blockers**  

**External dataset** for validating model's performance. (Consists of atoms that were not included in the training dataset).  

| Use | Both | Blocker | Non-blocker | Source |
|-----|------|---------|-------------|--------|
| | 3511 | 1773 | 1738 | Cai C, et al. |
| External dataset | 325 | 175 | 150 | Chen Y, et al. |
| | 408 | 227 | 181 | Karim A, et al. |

**Overview of the hERGAT model**

![Figure 2](https://github.com/user-attachments/assets/ea6bf039-1591-4909-b35c-165b8d14ada1)  

**Feature Extraction**  
1. **Graphical feature extraction** - **Atom connectivity**, **Bond types**, **Neighborhood relationship capabilities**.
   - Used **RdKit** library to acquire atom and bond features.
   - The **atom feature vector** consists of a `39-bit vector`.
   - The **Bond feature vector** has a `10-bit vector`.
   - Generated an **adjacency matrix** that considers the node's connections. (`1` signifies nodes are connected, `0` signifies they are not connected).
     
2. **Physicochemical properties extraction** - **Functions and Behaviours**
   Physicochemical properties taken into consideration for this study are..
   - **ALOGP** : (Octanol-water partition coefficienct : Lipid affinity of a compound).
   - **MW** : (Molecular Weight).
   - **TPSA** : (Topological polar surface-area).
   - **HBA** : (Hydrogen Bond acceptors).
   - **HBD** : (Hydrogen Bond donors).

- A high **ALOGP** indicates that the compound will be better absorbed in a lipid environment, correlates with its `affinity for the hERG channel`.
- Larger **MW** is related to an increased likelihood of `hERG channel inhibition`.
- **HBD** & **HBA** do not differ among hERG inhibitors but are useful in `distinguishing active hERG inhibitors`.
- **TPSA** has a `negative correlation` between `TPSA` and `hERG inhibition potency`.

![Research Paper ss01](https://github.com/user-attachments/assets/2b30e5d7-5ac7-4a06-b1f2-1624684d15dc)  


   












