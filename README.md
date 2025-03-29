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


## Outlining featurization process of hERG dataset using Ersilia representation model.  

I selected 'eos4u6p' (**Chemical checker signaturizer**) and implemented it in a python script. 

Explored [Ersilia Model Hub](https://ersilia.io/model-hub) for "Representation" models that convert SMILES into numerical features suitable for machine learning.  

Reviewed models like '**eos8ub5**' (Coconut projections, 8 dims), '**eos2db3**' (DrugBank projections, 8 dims), '**eos8f2t**' (Scaled WHALES, 11 dims), and '**eos4u6p**' (Chemical checker signaturizer, 3200 dims).  

**Model Chosen**  'eos4u6p' - "Chemical Checker 25 Bioactivity Signatures (2020-02 version)"  

It generates 25 signatures (128 dimensions each) totaling 3200 features per compound, capturing 2D/3D fingerprints, scaffolds, binding affinities, side effects and cell bioassay data.  

hERG cardiotoxicity depends on ion channel binding and side effects -- eos4u6p's signatures directl target these, unlike pure structural models (Eg. Coconut projections)  

3200 dimensions provide a detailed view making it ideal if i apply Random Forest.  (Helps my model uncover complex patterns in hERG blockage).  

'**eos4u6p**' has been trained on diverse bioactivity data making it suitable for hERG's synthetic drug-like compounds over natural product focused models.  

But, it was **computionally challenging** -- 41MB output. WSL Handled it (Took complete 45 mins though :( )  

**Implementing Featurization**  

I created a 'featurize.py' in 'notebooks/' folder.  

Combined 'hERG_train.csv', 'hERG_valid.csv' & 'hERG_test.csv' into 655 SMILES using **pandas**  

Used 'ErsiliaModel("eos4u6p") to fetch and serve model.  

Processed SMILES in batches of 20 to manage memory (as 3200 dims is huge).  

Saved features to 'data/hERG_ccsign_features.csv' - 655 rows, 3202 columns (3200 features + SMILES + Label).  

Please refer to '**notebooks/featurize.py**' for full implementation.  

**Ran in WSL Ubuntu with Conda environment 'ersilia'** (Also Python3, RDKit, pandas, numpy installed).  

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

**Visualizing raw hERG dataset**  

![hERG_label_dist](https://github.com/user-attachments/assets/7353466f-0b2d-4e66-ab15-8e055afb8309)  

Our data is not imbalanced (it has 70:30 ratio in every set) and we can train our models now.  

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

`notebooks/extract_smiles.py`  

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











