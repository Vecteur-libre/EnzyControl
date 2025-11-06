import argparse
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from data import utils as du
import os
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser(description="Enzyme filter script.")

parser.add_argument(
    "--metadata_dir",
    help="Path to metadata.csv",
    type=str,
    default="/data02/sc_data/enzymeframeflow/processed_enzyme/metadata.csv",
)

parser.add_argument(
    "--pkl_dir",
    help="Path to directory with processed pkl files.",
    type=str,
    default="/data02/sc_data/enzymeframeflow/processed_enzyme",
)

parser.add_argument("--num_chains", help="Number of chains", type=int, default=1)
parser.add_argument(
    "--min_res", help="Minimum number of residues", type=int, default=60
)
parser.add_argument(
    "--max_res", help="Maximum number of residues", type=int, default=512
)
parser.add_argument("--max_coil_percent", type=int, default=0.5)
parser.add_argument("--rog_quantile", type=int, default=0.96)


def _rog_filter(df, quantile):
    y_quant = pd.pivot_table(
        df,
        values="radius_gyration",
        index="modeled_seq_len",
        aggfunc=lambda x: np.quantile(x, quantile),
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    max_len = df.modeled_seq_len.max()
    pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

    row_rog_cutoffs = df.modeled_seq_len.map(lambda x: pred_y[x - 1])
    return df[df.radius_gyration < row_rog_cutoffs]


def filter_metadata(args, raw_csv):
    raw_csv = args.metadata_dir
    
    data_csv = pd.read_csv(raw_csv)

    ## only one chain
    data_csv = data_csv[data_csv.num_chains.isin([args.num_chains])]

    ## num_res: 60~512
    data_csv = data_csv[
        (data_csv.modeled_seq_len >= args.min_res)
        & (data_csv.modeled_seq_len <= args.max_res)
    ]

    ## max coil_percent:0.5
    data_csv = data_csv[data_csv.coil_percent <= args.max_coil_percent]

    ## rog_quantile: 0.96
    data_csv = _rog_filter(data_csv, args.rog_quantile)

    ## drop rows that lack EC_level3
    data_csv = data_csv.dropna(subset=["EC_level4"])
    
    
    ## add one additional column name sequence
    data_csv.insert(data_csv.shape[1],'sequence', np.nan)

    return data_csv


def get_sequence(pkl_dir:str):

    enzyme_feats = du.read_pkl(pkl_dir)
    
    ## enzyme sequence
    res_name = ''.join(enzyme_feats["res_name"])
    
    return res_name

def process_row(row):
    pkl_path = row['processed_path']  # Replace with actual column name
    sequence = get_sequence(pkl_path)
    row['sequence'] = sequence
    return row

def save_fasta(sequence_data, file_path):
    """
    Save sequences to a FASTA file.

    Parameters:
    sequence_data (list of tuples): Each tuple contains (name, sequence).
    file_path (str): Path to save the FASTA file.
    """
    with open(file_path, 'w') as fasta_file:
        for name, sequence in sequence_data:
            fasta_file.write(f">{name}\n{sequence}\n")


def process_metadata(metadata_filtered, output_dir):
    """
    Process metadata_filtered and save FASTA files based on EC levels.

    Parameters:
    metadata_filtered (pd.DataFrame): The filtered metadata with 'sequence' and EC levels.
    output_dir (str): Directory where the FASTA files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Track processed PDB codes to avoid duplication
    processed_pdbs = set()

    def process_group(group, level_name):
        """
        Helper function to process and save groups at any EC level.

        Parameters:
        group (pd.DataFrame): The group to process.
        level_name (str): The EC level name (e.g., EC_level3, EC_level2, EC_level1).

        Returns:
        list: List of PDB codes that were not saved due to single sequence.
        """
        remaining_pdbs = []
        grouped = group.groupby(level_name)

        for ec_level, sub_group in grouped:
            sequence_data = [
                (row['PDB code'], row['sequence'])
                for _, row in sub_group.iterrows()
                if pd.notnull(row['sequence'])
                and isinstance(row['sequence'], str)
                and row['PDB code'] not in processed_pdbs
            ]

            if len(sequence_data) > 1:
                # Save FASTA file if more than one sequence
                file_path = os.path.join(output_dir, f"{ec_level}.fasta")
                save_fasta(sequence_data, file_path)
                print(f"FASTA file saved: {file_path}")

                # Mark these sequences as processed
                processed_pdbs.update([name for name, _ in sequence_data])
            else:
                # Keep track of unsaved PDB codes
                remaining_pdbs.extend(sub_group['PDB code'].tolist())

        return remaining_pdbs

    # Process by EC_level4
    remaining_level4 = process_group(metadata_filtered, 'EC_level4')

    # Process by EC_level3 for remaining sequences
    remaining_data_level3 = metadata_filtered[metadata_filtered['PDB code'].isin(remaining_level4)]
    remaining_level3 = process_group(remaining_data_level3, 'EC_level3')

    # Process by EC_level2 for remaining sequences
    remaining_data_level2 = metadata_filtered[metadata_filtered['PDB code'].isin(remaining_level3)]
    remaining_level2 = process_group(remaining_data_level2, 'EC_level2')

    # Process by EC_level1 for remaining sequences
    remaining_data_level1 = metadata_filtered[metadata_filtered['PDB code'].isin(remaining_level2)]
    remaining_level1 = process_group(remaining_data_level1, 'EC_level1')

    # Report unsaved sequences
    if remaining_level1:
        print("Sequences that could not be saved to any group:")
        unsaved_data = metadata_filtered[metadata_filtered['PDB code'].isin(remaining_level1)]
        print(unsaved_data[['PDB code', 'EC_level4', 'EC_level3', 'EC_level2', 'EC_level1']])
        metadata_filtered = metadata_filtered[~metadata_filtered['PDB code'].isin(remaining_level1)]
    return metadata_filtered
        



def main():

    args = parser.parse_args()

    # Load and filter metadata
    metadata_filtered = filter_metadata(args, args.metadata_dir)

    with Pool(cpu_count()) as pool:
        rows=pool.map(process_row, [row for _, row in metadata_filtered.iterrows()])
    
    # Convert processed rows back to DataFrame
    metadata_filtered = pd.DataFrame(rows)

    metadata_filtered = metadata_filtered.drop_duplicates(subset='sequence')
    
    # Save the updated metadata
    output_path = os.path.join(os.path.dirname(args.metadata_dir), "metadata_with_sequence.csv")
    metadata_filtered.to_csv(output_path, index=False)
    print(f"Updated metadata saved to {output_path}")
    
    fasta_output_dir = os.path.join(os.path.dirname(args.metadata_dir), "fasta_files")
    
    metadata_filtered = process_metadata(metadata_filtered, fasta_output_dir)
    metadata_filtered.to_csv(output_path, index=False)
    print(f"Remaining sequences have been removed, updated metadata saved to {output_path}")

if __name__ =="__main__":
    main()