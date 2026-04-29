import argparse
import torch
from extract_PLIC import PLIC 
import numpy as np
from scipy import sparse
import os
import re
from collections import defaultdict
import shutil
from tqdm import tqdm
from utils.data import PDBProtein, parse_sdf_file

def load_item(protein_path):
    with open(protein_path, 'r') as f:
        pdb_block = f.read()
    return pdb_block

def extract_pocket_and_save(protein_block, row_ligand_fn, radius):
    protein = PDBProtein(protein_block)
    ligand = parse_sdf_file(row_ligand_fn)
    pdb_block_pocket = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand, radius)
    )
    # Remove element names at the end of each line
    lines = pdb_block_pocket.splitlines()
    trimmed_lines = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):    
            trimmed_lines.append(line[:66].rstrip())
        else:
            trimmed_lines.append(line)
    # Write to new file  
    pocket_fn = os.path.splitext(protein_fn)[0] + '_pocket%d.pdb' % radius

    with open(pocket_fn, 'w') as f:
        f.write('\n'.join(trimmed_lines) + '\n')
    return pocket_fn

def trans_np_to_tensor(pli):
    pli = [torch.from_numpy(arr) for arr in pli]
    pli = torch.cat(pli, dim=0)
    # pli = pli.to(torch.float32)
    return pli


def cosine_similarity(X, Y=None, dense_output=True):

    K = safe_sparse_dot(X, Y.T, dense_output=dense_output)

    return K[0][0]

def safe_sparse_dot(a, b, *, dense_output=False):
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()
    return ret

def L2_norm(pli):
    pli = np.array(pli)
    pli = pli / np.linalg.norm(pli, axis=1, keepdims=True)
    return pli


def group_sdf_files_by_guidance_scale(ligand_dir):
    # Create a default dict to store grouped files
    grouped_files = defaultdict(list)
    
    # Iterate through all files in ligand_dir
    for filename in os.listdir(ligand_dir):
        # Check if file ends with .sdf
        if filename.endswith('.sdf'):
            # Use regex to match group part in filename, allowing float numbers
            match = re.match(r'.*_(\d+\.\d+|\d+)\.sdf', filename)
            if match:
                group = match.group(1)
                # Convert group to float
                group_float = float(group)
                grouped_files[group_float].append(filename)
            else:
                print(f"Group not matched: {filename}")
    
    return grouped_files




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--row_ligand_fn', type=str, required=True)
    parser.add_argument('--ligand_dir', type=str, required=True)
    parser.add_argument('--protein_fn', type=str, required=True)
    parser.add_argument('--radius', type=int, default=10)
    args = parser.parse_args()

    plic = PLIC()
    row_ligand_fn = args.row_ligand_fn  
    protein_fn = args.protein_fn 
    ligand_dir = args.ligand_dir  

    grouped_files = group_sdf_files_by_guidance_scale(ligand_dir)
    sorted_grouped_files = {k: v for k, v in sorted(grouped_files.items(), key=lambda item: int(item[0]))}
    # Load structure
    protein_block = load_item(protein_fn)

    # Extract protein pocket
    pocket_fn = extract_pocket_and_save(protein_block, row_ligand_fn, args.radius)

    pli_row = plic.pli_processor(row_ligand_fn, pocket_fn)
    pli_row = pli_row[:, :-1]
    pli_row = pli_row.reshape(1, -1)
    pli_row = L2_norm(pli_row)


    average_similarity_gs_list = {}

    # Get parent directory of ligand_dir
    ligand_dir_parent = os.path.dirname(ligand_dir)
    # Save sorted_similarity_dict in specified folder
    out_dir_path = os.path.join(ligand_dir_parent, os.path.basename(ligand_dir) + '_similarity_output')
    os.makedirs(out_dir_path, exist_ok=True)

    for guidance_scale, ligand_fns in sorted_grouped_files.items():
        failed_num = 0
        failed_list = []
        similarity_dict = {}
        for i, ligand_fn in enumerate(tqdm(ligand_fns)):
            base_name = os.path.basename(ligand_fn)
            base_name = os.path.splitext(base_name)[0]
            ligand_fn = os.path.join(ligand_dir, ligand_fn)
            pli = plic.pli_processor(ligand_fn, pocket_fn)
            if pli is None:
                print(f"Failed to calculate PLI for {ligand_fn}")
                failed_num += 1
                failed_list.append(ligand_fn)
                continue
            pli= pli[:, :-1]
            pli = pli.reshape(1, -1)
            pli = L2_norm(pli)
            similarity = cosine_similarity(pli_row, pli)
            similarity_dict[f"{base_name}_similarity"] =  np.round(similarity, 4)

        sorted_similarity_dict = {k: v for k, v in sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)}
        successful_calc_count = len(similarity_dict)

        average_similarity = np.mean(list(similarity_dict.values()))
        average_similarity = np.round(average_similarity, 4)
        sorted_similarity_dict["average_similarity"] = average_similarity
        sorted_similarity_dict["successful_calculations_count"] = successful_calc_count
        average_similarity_gs_list[f'guidance_scale_{guidance_scale}'] = average_similarity

        output_fn = os.path.join(out_dir_path, f"{os.path.basename(ligand_dir)}_similarity_pli_only_gs_{guidance_scale}.pt")
        torch.save(sorted_similarity_dict, output_fn)

        k = 10
        top_k_similarity_dir = os.path.join(out_dir_path, f'top_{k}_similarity_gs_{guidance_scale}')

        os.makedirs(top_k_similarity_dir, exist_ok=True)

        top_k_similarities = list(sorted_similarity_dict.items())[:k]

        for name, similarity in top_k_similarities:
            if name.endswith('_similarity'):
                ligand_name = name.replace('_similarity', '')
                ligand_full_path = os.path.join(ligand_dir, ligand_name + '.sdf')
                if os.path.exists(ligand_full_path):
                    shutil.copy(ligand_full_path, top_k_similarity_dir)
                else:
                    print(f"File {ligand_full_path} does not exist, cannot copy.")

        print(f"Failed to calculate PLI for {failed_num} files: {failed_list}")
        print("")
        print(sorted_similarity_dict)
        print("")
        print(f"Average similarity: {average_similarity}")
    print(average_similarity_gs_list)
    
