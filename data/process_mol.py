import os
import pandas as pd
import numpy as np
import pickle
from unimol_tools import UniMolRepr
from tqdm import tqdm


clf = UniMolRepr(
    data_type='molecule',
    remove_hs=False,
    model_name='unimolv1', 
    model_size='84m',
)

csv_path = "/metadata/metadata_with_sequence.csv" 
df = pd.read_csv(csv_path)

for index, row in tqdm(df.iterrows(), total=len(df)):
    smiles = row['Canonical SMILES']
    protein_pkl_path = row['processed_path']
    
    try:
        smiles_list = [smiles]
        unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)

        protein_dir = os.path.dirname(protein_pkl_path)
        molecule_pkl_path = os.path.join(protein_dir, "molecule.pkl")

        with open(molecule_pkl_path, "wb") as f:
            pickle.dump(unimol_repr, f)

        print(f"Saved molecule embedding to {molecule_pkl_path}")

    except Exception as e:
        print(f"Error processing {smiles}: {e}")

print("Processing complete.")
