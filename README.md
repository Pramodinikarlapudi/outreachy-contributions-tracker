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

Please refer to https://youtu.be/YV74aapk72A?si=oWJm35VK9-bY9f7m, if you aren't able to follow these steps / face any issues.  

