import os
import numpy as np
import torch
import subprocess
import pandas as pd
import shutil
from biotite.sequence.io import fasta
from typing import Optional
import argparse
from analysis import metrics
from data import utils as du
from typing import Dict
import esm
import GPUtil
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# from ESP_code.ES_prediction import *
# Define the parser


class Enzyme_eval:
    def __init__(
        self,
        conf,
    ):
        """
        Initialize evaluation

        Args:
            conf : evaluation config.
        """
        self._log = logging.getLogger(__name__)
        self._conf = conf
        torch.hub.set_dir(self._conf.pt_hub_dir)

        if torch.cuda.is_available():
            if self._conf.gpu_id is None:
                available_gpus = "".join(
                    [str(x) for x in GPUtil.getAvailable(order="memory", limit=4)]
                )
                self.device = f"cuda:{available_gpus[0]}"

            else:
                self.device = f"cuda:{self._conf.gpu_id}"
        else:
            self.device = "cpu"
        self._log.info(f"Using device: {self.device}")
        # import pdb;pdb.set_trace()
        # self._folding_model = esm.pretrained.esmfold_v1().eval()
        # self._folding_model = self._folding_model.to(self.device)

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output

    def run_self_consistency(
        self,
        decoy_pdb_dir: str,
        reference_pdb_path: str,
        motif_mask: Optional[np.ndarray] = None,
    ):
        """Run self-consistency on design proteins against reference protein.

        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """

        # Run PorteinMPNN
        # import pdb;pdb.set_trace()
        ## output_path = /sample_7/self_consistency/parsed_pdbs.jsonl
        output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen(
            [
                "python",
                f"{self._conf._pmpnn_dir}/helper_scripts/parse_multiple_chains.py",
                f"--input_path={decoy_pdb_dir}",
                f"--output_path={output_path}",
            ]
        )
        _ = process.wait()
        num_tries = 0
        ret = -1
        pmpnn_args = [
            "python",
            f"{self._conf._pmpnn_dir}/protein_mpnn_run.py",
            "--out_folder",
            decoy_pdb_dir,
            "--jsonl_path",
            output_path,
            "--num_seq_per_target",
            str(self._conf.seq_per_sample),
            "--sampling_temp",
            "0.1",
            "--seed",
            "38",
            "--batch_size",
            "1",
        ]
        if self._conf.gpu_id is not None:
            pmpnn_args.append("--device")
            pmpnn_args.append(str(self._conf.gpu_id))
        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                self._log.info(f"Failed ProteinMPNN. Attempt {num_tries}/5")
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e
        ## mpnn_fasta_path = /sample_7/self_consistency/seqs/sample.fa
        mpnn_fasta_path = os.path.join(
            decoy_pdb_dir,
            "seqs",
            os.path.basename(reference_pdb_path).replace(".pdb", ".fa"),
        )
        # import pdb;pdb.set_trace()
        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        mpnn_results = {
            "tm_score": [],
            "sample_path": [],
            "header": [],
            "sequence": [],
            "rmsd": [],
        }
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            mpnn_results["motif_rmsd"] = []
        ## esmf_dir = sample_7/self_consistency/esmf, folding for each sequence of mpnn_fasta
        esmf_dir = os.path.join(decoy_pdb_dir, "esmf")
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        # import pdb;pdb.set_trace()
        sample_feats = du.parse_pdb_feats("sample", reference_pdb_path)
        for i, (header, string) in enumerate(fasta_seqs.items()):
            self._log.info(
                f"Processing sequence {i}: header={header}, length={len(string)}"
            )
            try:
                # Run ESMFold
                esmf_sample_path = os.path.join(esmf_dir, f"sample_{i}.pdb")
                _ = self.run_folding(string, esmf_sample_path)
                # import pdb;pdb.set_trace()
                esmf_feats = du.parse_pdb_feats("folded_sample", esmf_sample_path)
                sample_seq = du.aatype_to_seq(sample_feats["aatype"])

                # Calculate scTM of ESMFold outputs with reference protein
                _, tm_score = metrics.calc_tm_score(
                    sample_feats["bb_positions"],
                    esmf_feats["bb_positions"],
                    sample_seq,
                    sample_seq,
                )
                rmsd = metrics.calc_aligned_rmsd(
                    sample_feats["bb_positions"], esmf_feats["bb_positions"]
                )
                if motif_mask is not None:
                    sample_motif = sample_feats["bb_positions"][motif_mask]
                    of_motif = esmf_feats["bb_positions"][motif_mask]
                    motif_rmsd = metrics.calc_aligned_rmsd(sample_motif, of_motif)
                    mpnn_results["motif_rmsd"].append(motif_rmsd)
                mpnn_results["rmsd"].append(rmsd)
                mpnn_results["tm_score"].append(tm_score)
                mpnn_results["sample_path"].append(esmf_sample_path)
                mpnn_results["header"].append(header)
                mpnn_results["sequence"].append(string)
            except Exception as e:
                self._log.error(
                    f"Error processing sequence {i}: header={header}, error={e}"
                )
                continue  # Skip to the next sequence
        # import pdb;pdb.set_trace()
        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, "sc_results.csv")
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)

    def run_proteinMPNN_RF(self):
        for dirpath, dirnames, filenames in os.walk(self._conf.sample_dir):
            for dirname in dirnames:
                start_time = time.time()
                # if dirname.endswith("protein"):
                if dirname:
                    ### samples_for_mol: /data02/sc_data/enzymeframeflow/model_output/4keq
                    # import pdb;pdb.set_trace()
                    dirname = os.path.join(dirname,"rf_samples")
                    samples_for_mol = os.path.join(dirpath, dirname)
                    # samples_for_mol = os.path.join(dirpath, dirname, "rf_samples")

                    # subdirs = [
                    #     os.path.join(samples_for_mol, subdir)
                    #     for subdir in os.listdir(samples_for_mol)
                    #     if os.path.isdir(os.path.join(samples_for_mol, subdir))
                    #     and subdir.startswith("sample_")
                    # ]
                    subdirs = [os.path.join(samples_for_mol, subdir) for subdir in os.listdir(samples_for_mol) if subdir.startswith("sample")]
                    # import pdb;pdb.set_trace()
                    ## subdir = /data02/sc_data/enzymeframeflow/model_output/4keq/sample_7
                    self._log.info(f"Processing {dirname}")
                    for subdir in subdirs:

                        sc_output_dir = os.path.join(subdir, "self_consistency")
                        sc_results_path = os.path.join(sc_output_dir, "sc_results.csv")

                        if os.path.exists(sc_output_dir) and os.path.exists(
                            sc_results_path
                        ):
                            self._log.info(
                                f"Skipping {subdir}, sc_results.csv already exists."
                            )
                            continue

                        ## pdb_path = /data02/sc_data/enzymeframeflow/model_output/4keq_protein/sample_7/sample.pdb
                        pdb_path = os.path.join(subdir, "sample.pdb")
                        ## sc_output_dir = /data02/sc_data/enzymeframeflow/model_output/4keq_protein/sample_7/self_consistency
                        sc_output_dir = os.path.join(subdir, "self_consistency")
                        os.makedirs(sc_output_dir, exist_ok=True)
                        shutil.copy(
                            pdb_path,
                            os.path.join(sc_output_dir, os.path.basename(pdb_path)),
                        )
                        # import pdb;pdb.set_trace()
                        _ = self.run_self_consistency(
                            sc_output_dir, pdb_path, motif_mask=None
                        )
                elapsed_time = time.time() - start_time
                self._log.info(f"Finished {dirname} in {elapsed_time:.2f} seconds.")
                
    def run_proteinMPNN(self):
        for dirpath, dirnames, filenames in os.walk(self._conf.sample_dir):
            for dirname in dirnames:
                start_time = time.time()
                if dirname.endswith("protein"):
                    ### samples_for_mol: /data02/sc_data/enzymeframeflow/model_output/4keq
                    # import pdb;pdb.set_trace()
                    samples_for_mol = os.path.join(dirpath, dirname)
                    # samples_for_mol = os.path.join(dirpath, dirname, "rf_samples")

                    subdirs = [
                        os.path.join(samples_for_mol, subdir)
                        for subdir in os.listdir(samples_for_mol)
                        if os.path.isdir(os.path.join(samples_for_mol, subdir))
                        and subdir.startswith("sample_")
                    ]
                    # import pdb;pdb.set_trace()
                    ## subdir = /data02/sc_data/enzymeframeflow/model_output/4keq/sample_7
                    self._log.info(f"Processing {dirname}")
                    for subdir in subdirs:

                        sc_output_dir = os.path.join(subdir, "self_consistency")
                        sc_results_path = os.path.join(sc_output_dir, "sc_results.csv")

                        if os.path.exists(sc_output_dir) and os.path.exists(
                            sc_results_path
                        ):
                            self._log.info(
                                f"Skipping {subdir}, sc_results.csv already exists."
                            )
                            continue

                        ## pdb_path = /data02/sc_data/enzymeframeflow/model_output/4keq_protein/sample_7/sample.pdb
                        pdb_path = os.path.join(subdir, "sample.pdb")
                        ## sc_output_dir = /data02/sc_data/enzymeframeflow/model_output/4keq_protein/sample_7/self_consistency
                        sc_output_dir = os.path.join(subdir, "self_consistency")
                        os.makedirs(sc_output_dir, exist_ok=True)
                        shutil.copy(
                            pdb_path,
                            os.path.join(sc_output_dir, os.path.basename(pdb_path)),
                        )
                        # import pdb;pdb.set_trace()
                        _ = self.run_self_consistency(
                            sc_output_dir, pdb_path, motif_mask=None
                        )
                elapsed_time = time.time() - start_time
                self._log.info(f"Finished {dirname} in {elapsed_time:.2f} seconds.")
                
                
    def process_IF_results(self):
        """
        merge inverse folding results.
        """
        
        IF_results = []
        meta_data = pd.read_csv(self._conf.metadata)
        for dirpath, dirnames, filenames in os.walk(self._conf.sample_dir):
            for dirname in dirnames:
                if dirname.endswith("protein"):
                    ### samples_for_mol: /data02/sc_data/enzymeframeflow/model_output/4keq_protein
                    # import pdb;pdb.set_trace()
                    samples_for_mol = os.path.join(dirpath, dirname)
                    gt_protein_name = dirname.replace(
                        "_protein", "_protein_processed.pdb"
                    )
                    gt_ligand_name = dirname.replace("_protein", "_ligand.sdf")
                    ground_truth_path = os.path.join(samples_for_mol, "ground_truth")
                    gt_protein_path = os.path.join(ground_truth_path, gt_protein_name)
                    gt_ligand_path = os.path.join(ground_truth_path, gt_ligand_name)
                    gt_ligand_smiles = meta_data[meta_data["pdb_name"] == dirname][
                        "Canonical SMILES"
                    ].values[0]
                    subdirs = [
                        os.path.join(samples_for_mol, subdir)
                        for subdir in os.listdir(samples_for_mol)
                        if os.path.isdir(os.path.join(samples_for_mol, subdir))
                        and subdir.startswith("sample_")
                    ]
                    ## subdir = /data02/sc_data/enzymeframeflow/model_output/4keq_protein/sample_7
                    for subdir in subdirs:

                        sc_path = os.path.join(
                            subdir, "self_consistency/sc_results.csv"
                        )
                        sc_results = pd.read_csv(sc_path).drop(0)
                        sc_results["gt_protein"] = gt_protein_path
                        sc_results["gt_ligand"] = gt_ligand_path
                        sc_results["gt_ligand_smiles"] = gt_ligand_smiles

                        IF_results.append(sc_results)
        final_sc_results = pd.concat(IF_results, ignore_index=True).iloc[:, 1:]
        # import pdb;pdb.set_trace()
        f_path = os.path.join(self._conf.sample_dir, "final_sc_results.csv")
        final_sc_results.to_csv(f_path, index=False)

    def run_gnina_samples(self):
        sc_dir = os.path.join(self._conf.sample_dir, "final_sc_results.csv")
        sc_results = pd.read_csv(sc_dir)
        sc_results["Gnina_top1"] = None
        sc_results["Gnina_top3"] = None
        sc_results["Gnina_all"] = None
        for index, row in sc_results.iterrows():
            try:
                receptor = row["sample_path"].replace(
                    "/data02/sc_data/enzymeframeflow", "/root/enzyme"
                )
                ligand = row["gt_ligand"].replace(
                    "/data02/sc_data/enzymeframeflow", "/root/enzyme"
                )
                autobox_ligand = receptor
                output = receptor.replace(".pdb", "_docked.sdf")
                seed = str(0)
                gpu_id = str(self._conf.gpu_id)
                # import pdb;pdb.set_trace()
                gnina_command = [
                    "docker",
                    "exec",
                    "gnina",
                    "bash",
                    "-c",
                    f"'CUDA_VISIBLE_DEVICES={gpu_id} gnina -r {receptor} -l {ligand} --autobox_ligand {autobox_ligand} -o {output} --seed {seed}'",
                ]

                self._log.info(
                    f"Executing GNINA for row {index}: {' '.join(gnina_command)}"
                )

                process = subprocess.Popen(
                    " ".join(gnina_command),
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = process.communicate()
                # import pdb;pdb.set_trace()
                if process.returncode != 0:
                    raise RuntimeError(
                        f"GNINA execution failed for row {index}: {stderr.decode().strip()}"
                    )

                affinity_pattern = re.compile(
                    r"^\s*\d+\s+([-+]?\d+\.\d+)", re.MULTILINE
                )
                affinities = [
                    float(match) for match in affinity_pattern.findall(stdout.decode())
                ]
                if not affinities:
                    return None, None

                gnina_top1 = affinities[0]
                gnina_top3 = sum(affinities[:3]) / min(3, len(affinities))
                gnina_all = np.mean(affinities)

                sc_results.at[index, "Gnina_top1"] = gnina_top1
                sc_results.at[index, "Gnina_top3"] = gnina_top3
                sc_results.at[index, "Gnina_all"] = gnina_all
                self._log.info(
                    f"Row {index} completed successfully. Output saved to {output}"
                )

            except Exception as e:
                self._log.info(f"Error processing row {index}:{e}")
                continue

        updated_sc_dir = os.path.join(
            self._conf.sample_dir, "updated_final_sc_results.csv"
        )
        sc_results.to_csv(updated_sc_dir, index=False)
        self._log.info(f"Updated results saved to {updated_sc_dir}")

    def run_gnina_task(self, row, index, config, log):
        try:
            receptor = row["sample_path"].replace(
                "/data02/sc_data/enzymeframeflow", "/root/enzyme"
            )
            ligand = row["gt_ligand"].replace(
                "/data02/sc_data/enzymeframeflow", "/root/enzyme"
            )
            autobox_ligand = receptor
            output = receptor.replace(".pdb", "_docked.sdf")
            seed = str(0)
            gpu_id = str(config.gpu_id)

            gnina_command = [
                "docker",
                "exec",
                "gnina",
                "bash",
                "-c",
                f"'CUDA_VISIBLE_DEVICES={gpu_id} gnina -r {receptor} -l {ligand} --autobox_ligand {autobox_ligand} -o {output} --seed {seed}'",
            ]

            log.info(f"Executing GNINA for row {index}: {' '.join(gnina_command)}")
            process = subprocess.Popen(
                " ".join(gnina_command),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(
                    f"GNINA execution failed for row {index}: {stderr.decode().strip()}"
                )

            affinity_pattern = re.compile(r"^\s*\d+\s+([-+]?\d+\.\d+)", re.MULTILINE)
            affinities = [
                float(match) for match in affinity_pattern.findall(stdout.decode())
            ]

            if not affinities:
                return index, None, None, None

            gnina_top1 = affinities[0]
            gnina_top3 = sum(affinities[:3]) / min(3, len(affinities))
            gnina_all = np.mean(affinities)

            log.info(f"Row {index} completed successfully. Output saved to {output}")
            return index, gnina_top1, gnina_top3, gnina_all

        except Exception as e:
            log.info(f"Error processing row {index}: {e}")
            return index, None, None, None

    def run_gnina_samples_parallel(self):
        sc_dir = os.path.join(self._conf.sample_dir, "final_sc_results.csv")
        sc_results = pd.read_csv(sc_dir)
        sc_results["Gnina_top1"] = None
        sc_results["Gnina_top3"] = None
        sc_results["Gnina_all"] = None

        tasks = []
        with ThreadPoolExecutor(
            max_workers=8
        ) as executor:  # Adjust `max_workers` based on your system's resources
            for index, row in sc_results.iterrows():
                tasks.append(
                    executor.submit(
                        self.run_gnina_task, row, index, self._conf, self._log
                    )
                )

            for future in as_completed(tasks):
                index, gnina_top1, gnina_top3, gnina_all = future.result()
                if gnina_top1 is not None:
                    sc_results.at[index, "Gnina_top1"] = gnina_top1
                    sc_results.at[index, "Gnina_top3"] = gnina_top3
                    sc_results.at[index, "Gnina_all"] = gnina_all

        updated_sc_dir = os.path.join(
            self._conf.sample_dir, "updated_final_sc_results.csv"
        )
        sc_results.to_csv(updated_sc_dir, index=False)
        self._log.info(f"Updated results saved to {updated_sc_dir}")

    # ./maxcluster64bit -l ./test.txt ./test_all_by_all_lite -C 2 -in -Rl ./test_tm_results.txt -Tm 0.5 ./test_output.txt

    def run_gnina_task_gt(self, row, index, config, log):
        try:
            _receptor = row["gt_protein"]
            _ligand = row["gt_ligand"]

            new_receptor = re.sub(r"/s[1-4]/", f'/{"samples_with_esmf"}/', _receptor)
            new_ligand = re.sub(r"/s[1-4]/", f'/{"samples_with_esmf"}/', _ligand)

            receptor = new_receptor.replace(
                "/data02/sc_data/enzymeframeflow", "/root/enzyme"
            )
            ligand = new_ligand.replace(
                "/data02/sc_data/enzymeframeflow", "/root/enzyme"
            )
            autobox_ligand = receptor
            output = receptor.replace(".pdb", "_docked.sdf")
            seed = str(0)
            gpu_id = str(config.gpu_id)

            gnina_command = [
                "docker",
                "exec",
                "gnina",
                "bash",
                "-c",
                f"'CUDA_VISIBLE_DEVICES={gpu_id} gnina -r {receptor} -l {ligand} --autobox_ligand {autobox_ligand} -o {output} --seed {seed}'",
            ]

            log.info(f"Executing GNINA for row {index}: {' '.join(gnina_command)}")
            process = subprocess.Popen(
                " ".join(gnina_command),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(
                    f"GNINA execution failed for row {index}: {stderr.decode().strip()}"
                )

            affinity_pattern = re.compile(r"^\s*\d+\s+([-+]?\d+\.\d+)", re.MULTILINE)
            affinities = [
                float(match) for match in affinity_pattern.findall(stdout.decode())
            ]

            if not affinities:
                return index, None, None, None

            gnina_top1 = affinities[0]
            gnina_top3 = sum(affinities[:3]) / min(3, len(affinities))
            gnina_all = np.mean(affinities)

            log.info(f"Row {index} completed successfully. Output saved to {output}")
            return index, gnina_top1, gnina_top3, gnina_all

        except Exception as e:
            log.info(f"Error processing row {index}: {e}")
            return index, None, None, None

    def run_gnina_gt_parallel(self):
        sc_dir = os.path.join(self._conf.sample_dir, "updated_final_sc_results.csv")
        sc_results = pd.read_csv(sc_dir)
        sc_results = sc_results[["gt_protein", "gt_ligand"]]
        sc_results = sc_results.drop_duplicates(subset=["gt_protein"])
        sc_results["Gnina_top1"] = None
        sc_results["Gnina_top3"] = None
        sc_results["Gnina_all"] = None

        tasks = []
        with ThreadPoolExecutor(
            max_workers=8
        ) as executor:  # Adjust `max_workers` based on your system's resources
            for index, row in sc_results.iterrows():
                tasks.append(
                    executor.submit(
                        self.run_gnina_task_gt, row, index, self._conf, self._log
                    )
                )

            for future in as_completed(tasks):
                index, gnina_top1, gnina_top3, gnina_all = future.result()
                if gnina_top1 is not None:
                    sc_results.at[index, "Gnina_top1"] = gnina_top1
                    sc_results.at[index, "Gnina_top3"] = gnina_top3
                    sc_results.at[index, "Gnina_all"] = gnina_all

        updated_sc_dir = os.path.join(self._conf.sample_dir, "gt_docking_results.csv")
        sc_results.to_csv(updated_sc_dir, index=False)
        self._log.info(f"Updated results saved to {updated_sc_dir}")

    def compute_metrix(self):
        sc_dir = os.path.join(self._conf.sample_dir, "updated_final_sc_results.csv")
        gt_dir = os.path.join(self._conf.sample_dir, "gt_docking_results.csv")
        gt_data = pd.read_csv(gt_dir)
        gt_data = gt_data[['gt_protein', 'GT_top1', 'GT_top3', 'GT_all']]
        data = pd.read_csv(sc_dir)
        data = data.merge(gt_data, on='gt_protein', how='left')
        sc_dir2 = os.path.join(self._conf.sample_dir, "updated_final_sc_results_gt.csv")
        data.to_csv(sc_dir2, index=False)
        
        data_tm_score = data[data["tm_score"] > 0.5]
        TM_score = len(data_tm_score) / len(data)
        data_rmsd = data[data["rmsd"] < 2]
        rmsd = len(data_rmsd) / len(data)
        
        filtered_data = data[(data["tm_score"] > 0.5) & (data["rmsd"] < 2)]
        # designability
        designability = len(filtered_data) / len(data)

        gnina_filtered = data[data["Gnina_top1"] < data["GT_top1"]]
        specificity = len(gnina_filtered) / len(data)
        
        binding_score1 = np.mean(data["Gnina_top1"])
        binding_score3 = np.mean(data["Gnina_top3"])
        binding_score = np.mean(data["Gnina_all"])
        binding_score_gt1 = np.mean(data["GT_top1"])
        binding_score_gt3 = np.mean(data["GT_top3"])
        binding_score_gt = np.mean(data["GT_all"])
        
        self._log.info(f"0.5TM_score: {TM_score}")
        self._log.info(f"2rmsd: {rmsd}")
        self._log.info(f"Designability: {designability}")
        self._log.info(f"Specificity: {specificity}")
        self._log.info(f"binding_score1: {binding_score1}")
        self._log.info(f"binding_score3: {binding_score3}")
        self._log.info(f"binding_score: {binding_score}")
        self._log.info(f"binding_score_gt1: {binding_score_gt1}")
        self._log.info(f"binding_score_gt3: {binding_score_gt3}")
        self._log.info(f"binding_score_gt: {binding_score_gt}")
        
        return designability, specificity, binding_score, binding_score_gt


    def get_cluster(self):
        sc_dir = os.path.join(self._conf.sample_dir, "updated_final_sc_results.csv")
        txt_file = "/home/user12/se3_diffusion/Cluster_results/all_pdb_my.txt"
        
        df = pd.read_csv(sc_dir)
        sample_paths = df["sample_path"]
        with open(txt_file, 'w', encoding='utf-8') as f:
            for path in sample_paths:
                f.write(f"{path}\n")
        
    def get_seq(self):
        sc_dir = os.path.join(self._conf.sample_dir, "updated_final_sc_results.csv")
        fasta_file = "./Cluster_results/seq.fasta"
        df = pd.read_csv(sc_dir)
        seqs = df["sequence"]

        with open(fasta_file, 'w', encoding='utf-8') as f:
            for idx, sequence in enumerate(seqs, start=1):
                f.write(f">sequence_{idx}\n{sequence}\n")

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation script.")

    parser.add_argument("--gpu_id", help="GPU to use.", type=int, default=1)

    parser.add_argument(
        "--seq_per_sample", help="How many sequences to generate.", type=int, default=3
    )
    parser.add_argument(
        "--pt_hub_dir",
        help="Set pytorch hub dir.",
        type=str,
        default="/home/user12/.cache/torch/",
    )
    parser.add_argument(
        "--cluster_dir",
        help="Make cluster for samples.",
        type=str,
        default="/home/user12/se3_diffusion/Cluster_results/",
    )
    parser.add_argument(
        "--metadata",
        help="metadata.csv",
        type=str,
        default="/data02/sc_data/enzymeframeflow/metadata_with_sequence.csv",
    )
    parser.add_argument(
        "--sample_dir",
        help="The path to store sampling results.",
        type=str,
        # default="/data02/sc_data/enzymeframeflow/result20250114/samples_with_esmf/",
        # default="/data02/sc_data/enzymeframeflow/RFDiffusion_folder/",
        default="/data02/sc_data/enzymeframeflow/result20250411/",
    )

    parser.add_argument(
        "--_pmpnn_dir",
        help="The path to ProteinMPNN project",
        type=str,
        default="/home/user12/se3_diffusion/ProteinMPNN",
    )
    parser.add_argument(
        "--container_name", help="The name of container.", type=str, default="gnina"
    )
    parser.add_argument("--max_threads", help="num of workers.", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    conf = get_args()
    # import pdb;pdb.set_trace()
    logging.basicConfig(level=logging.INFO)
    compute_metric = Enzyme_eval(conf)

    # compute_metric.run_proteinMPNN()
    # compute_metric.process_IF_results()
    compute_metric.run_gnina_samples_parallel()
    # compute_metric.run_gnina_gt_parallel()
    # compute_metric.compute_metrix()
    # compute_metric.get_seq()
