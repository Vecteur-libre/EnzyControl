import pandas as pd
import tempfile
import os
from data import utils as du
from analysis import metrics

def rename_chain_to_A(pdb_path):

    try:
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".pdb")
        with open(pdb_path, 'r') as f_in, open(temp_file.name, 'w') as f_out:
            for line in f_in:
                if line.startswith(('ATOM', 'HETATM')) and len(line) >= 22:
                    line = line[:21] + 'A' + line[22:]
                f_out.write(line)
        return temp_file.name
    except Exception as e:
        print(f"[跳过] 重命名链时出错：{pdb_path}，错误：{e}")
        return None

df = pd.read_csv("/data02/sc_data/enzymeframeflow/result20250411/updated_final_sc_results.csv")

rmsd_list = []
skipped = 0

for idx, row in df.iterrows():
    real_path_raw = row["gt_protein"]
    sample_path_raw = row["sample_path"]

    try:
        real_path = rename_chain_to_A(real_path_raw)
        sample_path = rename_chain_to_A(sample_path_raw)

        if real_path is None or sample_path is None:
            skipped += 1
            continue

        real_feats = du.parse_pdb_feats("real", real_path)
        sample_feats = du.parse_pdb_feats("sample", sample_path)

        rmsd = metrics.calc_aligned_rmsd(
            sample_feats["bb_positions"], real_feats["bb_positions"]
        )
        rmsd_list.append(rmsd)

    except Exception as e:
        print(f"[跳过] 第{idx}行处理失败：{e}")
        skipped += 1
        continue
    finally:
        if real_path and os.path.exists(real_path):
            os.remove(real_path)
        if sample_path and os.path.exists(sample_path):
            os.remove(sample_path)

if rmsd_list:
    average_rmsd = sum(rmsd_list) / len(rmsd_list)
    print(f"Average RMSD: {average_rmsd:.4f}")
else:
    print("所有数据均处理失败。")

print(f"成功处理：{len(rmsd_list)} 条，跳过：{skipped} 条。")

