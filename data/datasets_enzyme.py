import abc
import numpy as np
import pandas as pd
import logging
import tree
import torch
import random
import os
import json
import subprocess
from torch.utils.data import Dataset
from data import utils as du
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


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


def _length_filter(data_csv, min_res, max_res):
    return data_csv[
        (data_csv.modeled_seq_len >= min_res) & (data_csv.modeled_seq_len <= max_res)
    ]


def _plddt_percent_filter(data_csv, min_plddt_percent):
    return data_csv[data_csv.num_confident_plddt > min_plddt_percent]


def _max_coil_filter(data_csv, max_coil_percent):
    return data_csv[data_csv.coil_percent <= max_coil_percent]


def _process_csv_row(processed_file_path, f_sites):
    
    
    processed_feats = du.read_pkl(processed_file_path)

    protein_dir = os.path.dirname(processed_file_path)
    molecule_pkl_path = os.path.join(protein_dir, "molecule.pkl")
    substrate_feats = du.read_pkl(molecule_pkl_path)
    substrate = substrate_feats['atomic_reprs']
    substrate = torch.tensor(substrate)
    substrate = substrate.squeeze(0)

    current_len = substrate.shape[0]

    padded_substrate = torch.zeros(128, substrate.shape[-1])

    if current_len < 128:
        padded_substrate[:current_len, :] = substrate
    else:
        padded_substrate[:] = substrate[:128, :]

    substrate_mask = torch.zeros(128, dtype=torch.bool)
    substrate_mask[:current_len] = 1
    
    
    processed_feats["func_mask"] = np.zeros(len(processed_feats["bb_mask"]), dtype=bool)
    for index in f_sites:
        if 1 <= index <= len(processed_feats["bb_mask"]):
            processed_feats["func_mask"][index - 1] = True

    processed_feats = du.parse_chain_feats(
        processed_feats
    )  ## center bb pos and scale them, then update atom pos and bb pos. did not add extra features.

    # Only take modeled residues.
    modeled_idx = processed_feats[
        "modeled_idx"
    ]  ## include valid bb type, remove bb that does not belong to the 20 types.
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)
    del processed_feats["modeled_idx"]
    processed_feats = tree.map_structure(
        lambda x: x[min_idx : (max_idx + 1)], processed_feats
    )

    # Run through OpenFold data transforms.
    chain_feats = {
        "aatype": torch.tensor(processed_feats["aatype"]).long(),
        "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
    }

    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_gt_frames"])[
        :, 0
    ]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    res_plddt = processed_feats["b_factors"][:, 1]
    res_mask = torch.tensor(processed_feats["bb_mask"]).int()
    # pocket_mask = torch.tensor(processed_feats["pocket_mask"]).int()
    func_mask = torch.tensor(processed_feats["func_mask"]).int()

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    chain_idx = processed_feats["chain_index"]
    res_idx = processed_feats["residue_index"]
    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)
    all_chain_idx = np.unique(chain_idx).tolist()
    shuffled_chain_idx = (
        np.array(random.sample(all_chain_idx, len(all_chain_idx)))
        - np.min(all_chain_idx)
        + 1
    )

    for i, chain_id in enumerate(all_chain_idx):
        chain_mask = (chain_idx == chain_id).astype(int)
        chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
        new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

        # Shuffle chain_index
        replacement_chain_id = shuffled_chain_idx[i]
        new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask
    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(f"Found NaNs in {processed_file_path}")
    return {
        "res_plddt": res_plddt,
        "aatype": chain_feats["aatype"],
        "rotmats_1": rotmats_1,
        "trans_1": trans_1,
        "res_mask": res_mask,
        "chain_idx": new_chain_idx,
        "res_idx": new_res_idx,
        "func_mask": func_mask,
        "substrate":padded_substrate,
        "mol_mask": substrate_mask,
    }


def _add_plddt_mask(feats, plddt_threshold):
    feats["plddt_mask"] = torch.tensor(feats["res_plddt"] > plddt_threshold).int()


class BaseDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_cfg,
        is_training,
        task,
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv["aug_msa"] = metadata_csv.apply(
            lambda row: self.augment_msa(row["seq_len"], row["conserved_regions"]),
            axis=1,
        )
        metadata_csv = metadata_csv.sort_values("modeled_seq_len", ascending=False)
        # self.create_ec_split(metadata_csv)
        self.create_cluster_split(metadata_csv)
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    def __len__(self):
        return len(self.csv)

    @abc.abstractmethod
    def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        pass

    def _create_split(self, data_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = data_csv
            self._log.info(f"Training: {len(self.csv)} examples")
        else:
            if self._dataset_cfg.max_eval_length is None:
                eval_lengths = data_csv.modeled_seq_len
            else:
                eval_lengths = data_csv.modeled_seq_len[
                    data_csv.modeled_seq_len <= self._dataset_cfg.max_eval_length
                ]
            all_lengths = np.sort(eval_lengths.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths
            )
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = data_csv[data_csv.modeled_seq_len.isin(eval_lengths)]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby("modeled_seq_len").sample(
                self.dataset_cfg.samples_per_eval_length, replace=True, random_state=123
            )
            eval_csv = eval_csv.sort_values("modeled_seq_len", ascending=False)
            self.csv = eval_csv
            self._log.info(
                f"Validation: {len(self.csv)} examples with lengths {eval_lengths}"
            )
        self.csv["index"] = list(range(len(self.csv)))

    def create_ec_split(self, data_csv, rate=0.05):

        group_sizes = data_csv.groupby("EC_level2").size()
        eval_group_sizes = (group_sizes * rate).round().astype(int)

        eval_group_sizes = eval_group_sizes.clip(lower=1)

        eval_csv = (
            data_csv.groupby("EC_level2")
            .apply(
                lambda group: group.sample(
                    n=eval_group_sizes[group.name], random_state=123
                )
            )
            .reset_index(drop=True)
        )

        train_csv = data_csv.loc[~data_csv.index.isin(eval_csv.index)].reset_index(
            drop=True
        )

        with open("./metadata/train.fasta", "w") as f:
            for _, row in train_csv.iterrows():
                f.write(f">{row['PDB code']}\n{row['sequence']}\n")

        with open("./metadata/eval.fasta", "w") as f:
            for _, row in eval_csv.iterrows():
                f.write(f">{row['PDB code']}\n{row['sequence']}\n")

        cmd = [
            "cd-hit-2d",
            "-i",
            "./metadata/train.fasta",
            "-i2",
            "./metadata/eval.fasta",
            "-o",
            "./metadata/eval2novel.fasta",
            "-c",
            "0.9",
            "-n",
            "5",
            "-M",
            "16000",
            "-T",
            "8",
        ]
        subprocess.run(cmd, check=True)

        eval_pdb_codes = set(eval_csv["PDB code"])
        remaining_pdb_codes = set()
        with open("./metadata/eval2novel.fasta", "r") as f:
            for line in f:
                if line.startswith(">"):
                    remaining_pdb_codes.add(line.strip()[1:])
        removed_pdb_codes = eval_pdb_codes - remaining_pdb_codes

        removed_rows = eval_csv[eval_csv["PDB code"].isin(removed_pdb_codes)]
        eval_csv = eval_csv[~eval_csv["PDB code"].isin(removed_pdb_codes)]
        train_csv = pd.concat([train_csv, removed_rows])

        # for -n, we recommand 5 corresponding to 0.9 similarity
        # cd-hit-2d -i trainset -i2 testset -o testset2novel -c 0.9 -n 5 -d 0 -M 16000 -T 8

        train_csv.to_csv(self._dataset_cfg.trainset_dir)
        eval_csv.to_csv(self._dataset_cfg.testset_dir)

        if self.is_training:
            self.csv = train_csv
            self._log.info(f"Training: {len(self.csv)} examples")
        else:
            self.csv = eval_csv
            self._log.info(f"Validation: {len(self.csv)} examples")

        self.csv["index"] = list(range(len(self.csv)))

    def create_cluster_split(self, data_csv, rate=0.05,seed=123):
        
        random.seed(seed)
        
        with open("./metadata/all.fasta","w") as f:
            for _, row in data_csv.iterrows():
                f.write(f">{row['PDB code']}\n{row['sequence']}\n")

        cmd = [
            "cd-hit",
            "-i", "./metadata/all.fasta",
            "-o", "./metadata/clustered.fasta",
            "-c", "0.9",
            "-n", "5"
        ]

        subprocess.run(cmd, check=True)
        
        clusters = {}
        cluster_id = None 
        with open("./metadata/clustered.fasta.clstr", "r") as f:
            for line in f:
                if line.startswith(">Cluster"):
                    cluster_id = int(line.strip().split()[1])
                    clusters[cluster_id] = []
                elif ">" in line:
                    pdb_code = line.split(">")[1].split("...")[0]
                    clusters[cluster_id].append(pdb_code)
        
        cluster_ids = list(clusters.keys())
        random.shuffle(cluster_ids)
        test_size = int(len(cluster_ids) * rate)
        test_clusters = cluster_ids[:test_size]
        train_clusters = cluster_ids[test_size:]
        
        train_pdb_codes = [pdb for cid in train_clusters for pdb in clusters[cid]]
        test_pdb_codes = [pdb for cid in test_clusters for pdb in clusters[cid]]
        
        data = data_csv.set_index("PDB code")
        train_df = data.loc[train_pdb_codes].reset_index()
        test_df = data.loc[test_pdb_codes].reset_index()        

        train_df.to_csv(self._dataset_cfg.trainset_dir)
        test_df.to_csv(self._dataset_cfg.testset_dir)

        if self.is_training:
            self.csv = train_df
            self._log.info(f"Training: {len(self.csv)} examples")
        else:
            self.csv = test_df
            self._log.info(f"Validation: {len(self.csv)} examples")

        self.csv["index"] = list(range(len(self.csv)))
        
        
    def process_csv_row(self, csv_row):

        path = csv_row["processed_path"]
        seq_len = csv_row["modeled_seq_len"]
        f_sites = csv_row["aug_msa"]
        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        processed_row = _process_csv_row(path, f_sites)
        if use_cache:
            self._cache[path] = processed_row
        return processed_row

    def augment_msa(self, seq_len, conserved_regions):

        if (
            conserved_regions is None
            or pd.isna(conserved_regions)
            or conserved_regions == "[]"
        ):
            conserved_regions = []
        else:
            conserved_regions = (
                eval(conserved_regions)
                if isinstance(conserved_regions, str)
                else conserved_regions
            )

        num_res = seq_len
        conserved_set = set(conserved_regions)
        valid_indices = [i for i in range(num_res) if i not in conserved_set]

        if len(conserved_regions) >= num_res / 10:
            return conserved_regions

        min_motif_size = int(self._dataset_cfg.min_motif_percent * num_res)
        max_motif_size = int(self._dataset_cfg.max_motif_percent * num_res)
        total_motif_size = self._rng.integers(low=min_motif_size, high=max_motif_size)

        num_motifs = self._rng.integers(low=1, high=total_motif_size)

        attempt = 0
        while attempt < 100:
            motif_lengths = np.sort(
                self._rng.integers(low=1, high=max_motif_size, size=num_motifs)
            )
            cumulative_lengths = np.cumsum(motif_lengths)
            motif_lengths = motif_lengths[cumulative_lengths < total_motif_size]
            if len(motif_lengths) > 0:
                break
            attempt += 1
        if len(motif_lengths) == 0:
            motif_lengths = [total_motif_size]

        motif_mask = torch.zeros(num_res)
        seed_residues = self._rng.choice(
            valid_indices, size=len(motif_lengths), replace=False
        )
        for seed, motif_len in zip(seed_residues, motif_lengths):
            for i in range(motif_len):
                pos = seed + i
                if pos >= num_res or pos in conserved_set:
                    break
                motif_mask[pos] = 1.0

        augmented_indices = list(torch.where(motif_mask == 1)[0].numpy())
        return sorted(set(augmented_indices + conserved_regions))

    def get_motif_scaffold_mask(self, batch):
        """
        get motifs need to be generated,return scaffold (first one) and motif (second one)
        """
        motif_mask = batch["pocket_mask"]
        scaffold_mask = 1 - motif_mask
        return scaffold_mask * batch["res_mask"], motif_mask * batch["res_mask"]

    def get_aug_motif_mask(self, batch, rng):
        """
        return aug_conserved sequence, which will be the functional condition.
        """
        
        
        
        
        trans_1 = batch["trans_1"]
        num_res = trans_1.shape[0]

        min_motif_size = int(self._dataset_cfg.min_motif_percent * num_res)
        max_motif_size = int(self._dataset_cfg.max_motif_percent * num_res)

        # Sample the total number of residues that will be used as the aug_motif.
        total_motif_size = self._rng.integers(low=min_motif_size, high=max_motif_size)

        # Sample motifs at different locations.
        num_motifs = rng.integers(low=1, high=total_motif_size)

        # Attempt to sample
        attempt = 0
        while attempt < 100:
            # Sample lengths of each motif.
            motif_lengths = np.sort(
                rng.integers(low=1, high=max_motif_size, size=(num_motifs,))
            )

            # Truncate motifs to not go over the motif length.
            cumulative_lengths = np.cumsum(motif_lengths)
            motif_lengths = motif_lengths[cumulative_lengths < total_motif_size]
            if len(motif_lengths) == 0:
                attempt += 1
            else:
                break
        if len(motif_lengths) == 0:
            motif_lengths = [total_motif_size]

        # Ensure valid sampling positions do not overlap with pocket.
        invalid_residues = batch["pocket_mask"]  # Non-pocket positions
        valid_indices = torch.where(invalid_residues == 0)[0]  # Get valid positions
        if len(valid_indices) < total_motif_size:
            raise ValueError("Valid positions are insufficient for the motif size.")
        seed_residues = rng.choice(
            valid_indices.numpy(), size=(len(motif_lengths),), replace=False
        )

        # Construct the motif mask.
        motif_mask = torch.zeros(num_res)
        for motif_seed, motif_len in zip(seed_residues, motif_lengths):
            for i in range(motif_len):
                pos = motif_seed + i
                if pos >= num_res or batch["pocket_mask"][pos] == 1:
                    break
                motif_mask[pos] = 1.0
        return motif_mask * batch["res_mask"]

    # def setup_inpainting(self, feats, rng):
    #     diffuse_mask = self._sample_scaffold_mask(feats, rng)
    #     if 'plddt_mask' in feats:
    #         diffuse_mask = diffuse_mask * feats['plddt_mask']
    #     if torch.sum(diffuse_mask) < 1:
    #         # Should only happen rarely.
    #         diffuse_mask = torch.ones_like(diffuse_mask)
    #     feats['diffuse_mask'] = diffuse_mask

    def setup_inpainting_enzyme(self, feats, rng):
        # diffusion_mask: pocket index, need to be mask
        # 
        # _, diffuse_mask = self.get_motif_scaffold_mask(feats) 
        # aug_motif_mask = self.get_aug_motif_mask(feats, rng)
        
        diffuse_mask  = feats['pocket_mask'] * feats['res_mask']
        
        func_mask = feats['func_mask'] * feats['res_mask']
        if "plddt_mask" in feats:
            diffuse_mask = diffuse_mask * feats["plddt_mask"]
        if torch.sum(diffuse_mask) < 1:
            # Should only happen rarely.
            diffuse_mask = torch.ones_like(diffuse_mask)
        feats["diffuse_mask"] = diffuse_mask
        feats["func_mask"] = func_mask

    def __getitem__(self, row_idx):
        # Process data example.
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row)

        if self._dataset_cfg.add_plddt_mask:
            _add_plddt_mask(feats, self._dataset_cfg.min_plddt_threshold)
        else:
            feats["plddt_mask"] = torch.ones_like(feats["res_mask"])

        if self.task == "hallucination":
            """
            motif guidance
            """
            feats["diffuse_mask"] = torch.ones_like(feats["res_mask"]).bool()
        elif self.task == "inpainting":
            """
            motif amortization
            """
            if self._dataset_cfg.inpainting_percent < random.random():
                feats["diffuse_mask"] = torch.ones_like(feats["res_mask"])
            else:
                # rng = self._rng if self.is_training else np.random.default_rng(seed=123)
                # self.setup_inpainting(feats, rng)
                # # Center based on motif locations
                # motif_mask = 1 - feats['diffuse_mask']
                # trans_1 = feats['trans_1']
                # motif_1 = trans_1 * motif_mask[:, None]
                # motif_com = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1)
                # trans_1 -= motif_com[None, :]
                # feats['trans_1'] = trans_1
                rng = self._rng if self.is_training else np.random.default_rng(seed=123)
                self.setup_inpainting_enzyme(feats, rng)

                conserved_function = feats["func_mask"]

                feats["diffuse_mask"] = 1 - conserved_function
                # scaffold_mask = 1 - feats["diffuse_mask"]
                trans_1 = feats["trans_1"]
                conserved_function_1 = trans_1 * conserved_function[:, None]
                conserved_function_com = torch.sum(conserved_function_1, dim=0) / (
                    torch.sum(conserved_function) + 1
                )
                trans_1 -= conserved_function_com[None, :]
                feats["trans_1"] = trans_1
        else:
            raise ValueError(f"Unknown task {self.task}")
        feats["diffuse_mask"] = feats["diffuse_mask"].int()

        # Storing the csv index is helpful for debugging.
        feats["csv_idx"] = torch.ones(1, dtype=torch.long) * row_idx
        return feats


class EnzymeDataset(BaseDataset):

    def __init__(
        self,
        *,
        dataset_cfg,
        is_training,
        task,
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        self._cache = {}
        # self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

        # Process clusters
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        # import pdb;pdb.set_trace()
        # metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = self.raw_csv
        metadata_csv["aug_msa"] = metadata_csv.apply(
            lambda row: self.augment_msa(row["seq_len"], row["conserved_regions"]),
            axis=1,
        )
        metadata_csv = metadata_csv.sort_values("modeled_seq_len", ascending=False)

        metadata_csv = metadata_csv.sort_values("modeled_seq_len", ascending=False)

        self.create_cluster_split(metadata_csv)
        self._all_clusters = dict(enumerate(self.csv["EC_level2"].unique().tolist()))
        self._num_clusters = len(self._all_clusters)

    def _filter_metadata(self, raw_csv):
        """Filter metadata."""
        filter_cfg = self.dataset_cfg.filter
        ## only one chain
        data_csv = data_csv[data_csv.num_chains.isin(filter_cfg.num_chains)]

        ## num_res: 60~512
        data_csv = _length_filter(
            data_csv, filter_cfg.min_num_res, filter_cfg.max_num_res
        )

        ## max coil_percent: 0.5
        data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)

        ## rog_quantile: 0.96
        data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)

        ## delete rows that lack EC_level4
        data_csv = data_csv.dropna(subset=["EC_level4"])

        return data_csv

