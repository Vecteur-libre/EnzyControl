import os
import gzip
from datetime import datetime

def extract_chain_a_protein_from_gz(gz_path, output_path):
    atom_lines = []
    with gzip.open(gz_path, "rt") as infile:
        for line in infile:
            if line.startswith("ATOM") and line[21] == "A":
                atom_lines.append(line)

    if atom_lines:
        last_line = atom_lines[-1]
        last_serial = int(last_line[6:11])
        last_res_name = last_line[17:20]
        last_chain_id = last_line[21]
        last_res_seq = last_line[22:26]

        with open(output_path, "w") as outfile:
            outfile.writelines(atom_lines)
            ter_line = f"TER   {last_serial+1:5d}      {last_res_name} {last_chain_id}{last_res_seq}                                                      \n"
            outfile.write(ter_line)
            outfile.write("END\n")
        return True
    else:
        return False

def process_all_pdb_gz(input_dir, output_base_dir):
    os.makedirs(output_base_dir, exist_ok=True)
    log_path = os.path.join(input_dir, "processing_log.txt")

    with open(log_path, "w") as log:
        log.write(f"❗ 未成功处理的条目 - {datetime.now()}\n")
        log.write("=" * 50 + "\n")

        for filename in os.listdir(input_dir):
            if filename.endswith(".pdb.gz"):
                pdb_id = filename.replace(".pdb.gz", "")
                gz_path = os.path.join(input_dir, filename)

                output_dir = os.path.join(output_base_dir, pdb_id)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{pdb_id}_protein_processed.pdb")

                try:
                    success = extract_chain_a_protein_from_gz(gz_path, output_file)
                    if not success:
                        msg = f"{pdb_id}：没有链A的ATOM记录\n"
                        log.write(msg)
                        print(f"⚠️  {msg.strip()}")
                except Exception as e:
                    msg = f"{pdb_id}：处理异常 → {str(e)}\n"
                    log.write(msg)
                    print(f"💥  {msg.strip()}")

        log.write("=" * 50 + "\n")



# 使用方式示例（替换为你的真实路径）：
# process_all_pdb_gz("/path/to/your/folder")


# 示例调用（改成你的目录）
# process_all_pdb_gz("/path/to/your/folder")

if __name__ == "__main__":
    input_path = "/data02/sc_data/LiLei"
    output_path = "/data02/sc_data/LiLei/processed/"
    process_all_pdb_gz(input_path, output_path)