import pandas as pd
from chembl_websource_client.new_client import new_client

target = new_client.target
activity = new_client.activity

herg_targets = target.search('hERG').filter(organisms='Homo sapiens')
herg_chembl_id = herg_targets[0]['target_chembl_id']

activities_df = activities_df[['molecule_chembl_id', 'canonical_smiles', 'standard_value']]
activities_df = activities_df.dropna(subset=['canonical_smiles', 'standard_value'])
activities_df['standard_value'] = pd.to_numeric(activities_df['standard_value'], errors='coerce')
activities_df = activities_df.dropna(subset=['standard_value'])

activities_df['Label'] = activities_df['standard_value'].apply(lambda x: 1 if x < 10000 else 0)

activities_df = activities_df.drop_duplicates(subset=['canonical_smiles'], keep='first')

activities_df.to_csv('/mnt/d/outreachy-contributions-tracker/data/hERG_chembl_unseen.csv', index=False)

print(f"Saved {len(activities_df)} compounds to hERG_chembl_unseen.csv")
