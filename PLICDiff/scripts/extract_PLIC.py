from rdkit import Chem
import os
import traceback
import tempfile
import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool

import torch
from torch.utils.data import Subset
from plip.structure.preparation import PDBComplex
from datasets import PocketLigandPairDataset

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBIO import Select
from rdkit import Chem, RDLogger
import tempfile
from scipy.spatial import distance_matrix

INTERACTION_TYPES = ["pipi", "anion", "cation", "hbd", "hba", "hydro"]



class PLIC:
    def __init__(
        self,
        seed=2024
    ):
        
        self.seed = seed
        # self.lmdb_path = lmdb_path
        # self.split_data_path = split_data_path
        self._tmp_dir = '/home/swhuang/DL_code/PlicDiff_1/tmp'
        self.check_and_create_directory(self._tmp_dir)



    def check_and_create_directory(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Directory '{directory_path}' already exists.")


    def rd_file(self, fn):
        extension = fn.split(".")[-1]
        if extension == "sdf":
            mol = Chem.SDMolSupplier(fn)[0]
        elif extension == "mol2":
            mol = Chem.MolFromMol2File(fn)
        elif extension == "pdb":
            mol = Chem.MolFromPDBFile(fn)

        else:
            # print("Wrong file format...")
            return
        if mol is None:
            # print("No mol from file...")
            return
        return mol

    def _extract_binding_pocket(     
            self,
            ligand_mol,
            protein_pdb,
            cutoff=10,
            use_whole_protein=False,
        ):
        parser = PDBParser()
        if not os.path.exists(protein_pdb):
            return
        structure = parser.get_structure("protein", protein_pdb)
        ligand_positions = ligand_mol.GetConformer().GetPositions()
        class NonHeteroSelect(Select):
            def accept_residue(self, residue):
                if residue.get_resname() == "HOH":
                    return 0
                if residue.get_id()[0] != " ":
                    return 0
                else:
                    return 1

        class DistSelect(Select):
            def accept_residue(self, residue):
                if residue.get_resname() == "HOH":
                    return 0
                if residue.get_id()[0] != " ":
                    return 0
                residue_positions = np.array(
                    [
                        np.array(list(atom.get_vector()))
                        for atom in residue.get_atoms()
                        if "H" not in atom.get_id()
                    ]
                )

                min_dis = np.min(distance_matrix(residue_positions, ligand_positions))
                if min_dis < cutoff:
                    return 1
                else:
                    return 0


        io = PDBIO()
        io.set_structure(structure)
        fd, path = tempfile.mkstemp(
            suffix=".pdb", prefix="tmp_poc_", dir=self._tmp_dir
        )
        if use_whole_protein:
            io.save(path, NonHeteroSelect())
        else:
            io.save(path, DistSelect())
        m2 = Chem.MolFromPDBFile(path,removeHs=False)
        print(f'num_pocket is {m2.GetNumAtoms()}')
        structure2 = parser.get_structure("pocket", path)
        os.close(fd)
        return m2, structure2, path  

    def _filter_ligand(self, ligand_mol):
        if ligand_mol is None:
            print("ligand_mol is None")
            return False
        return True

    def _filter_pocket(self, pocket_mol):
        if pocket_mol is None:
            print("pocket_mol is None")
            return False
        return True
    
    def _join_complex(self, ligand_fn, pocket_fn, complex_fn=None):
        if complex_fn is None:
            fd, complex_fn = tempfile.mkstemp(
                suffix=".pdb", prefix="tmp_com_", dir=self._tmp_dir
            )
        command = f"obabel {ligand_fn} {pocket_fn} -O {complex_fn} -j -d 2> NUL"
        os.system(command)
        with open(complex_fn, "r") as f:
            lines = f.readlines()
        num_ligand_atom = Chem.SDMolSupplier(ligand_fn)[0].GetNumAtoms()
        new_lines = []
        for i, line in enumerate(lines):
            if i > 1 and i < num_ligand_atom + 2:
                new_line = (
                    line[:17] + "LIG" + line[20:25] + "1 " + line[27:]
                )  # enforce lig_resname as LIG
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        with open(complex_fn, "w") as f:
            f.writelines(new_lines)
        complex_mol = Chem.MolFromPDBFile(complex_fn)
        print(f'complex_atm_num={complex_mol.GetNumAtoms()}')
        return complex_mol, complex_fn, fd


    def _get_complex_interaction_info(   
        self,
        complex_fn,
    ):
        my_mol = PDBComplex()
        my_mol.load_pdb(complex_fn)
        ligs = [
            ":".join([x.hetid, x.chain, str(x.position)])
            for x in my_mol.ligands
            if x.hetid == "LIG"
        ]
        if len(ligs) == 0:
            return
        my_mol.analyze()
        my_interactions = my_mol.interaction_sets[ligs[0]] 

        anions = my_interactions.saltbridge_pneg
        cations = my_interactions.saltbridge_lneg
        hbds = my_interactions.hbonds_pdon
        hbas = my_interactions.hbonds_ldon
        hydros = my_interactions.hydrophobic_contacts
        pipis = my_interactions.pistacking

        # 1. salt-bridges
        anion_indices, cation_indices = [], []
        for an in anions:
            anion_indices += [x - 1 for x in an.negative.atoms_orig_idx]
        for ct in cations:
            cation_indices += [x - 1 for x in ct.positive.atoms_orig_idx]

        # 2. hydrogen bonds
        hbd_indices, hba_indices = [], []
        for hbd in hbds:
            hbd_indices += [hbd.d_orig_idx - 1]
        for hba in hbas:
            hba_indices += [hba.a_orig_idx - 1]

        # 3. Hydrophobic interactions
        hyd_indices = []
        for hyd in hydros:
            hyd_indices += [hyd.bsatom_orig_idx - 1]

        # 4. Pi-Pi Stackings
        pipi_indices = []
        for pi in pipis:
            pipi_indices += [x - 1 for x in pi.proteinring.atoms_orig_idx]

        anion_indices = list(set(anion_indices))
        cation_indices = list(set(cation_indices))
        hbd_indices = list(set(hbd_indices))
        hba_indices = list(set(hba_indices))
        hyd_indices = list(set(hyd_indices))
        pipi_indices = list(set(pipi_indices))

        return (
            anion_indices,
            cation_indices,
            hbd_indices,
            hba_indices,
            hyd_indices,
            pipi_indices,
            None,
        )
    
    def _get_one_hot_vector(self, item, item_list, use_unk=True):
        if item not in item_list:
            if use_unk:
                ind = -1
            else:
                print(f"Item not in the list: {item}")
                exit()
        else:
            ind = item_list.index(item) 
        if use_unk:
            return list(np.eye(len(item_list) + 1)[ind])
        else:
            return list(np.eye(len(item_list))[ind]) 

    def _get_pocket_interaction_matrix(self, ligand_n, pocket_n, info):
        anion, cation, hbd, hba, hydro, pipi, mask = info  
        pocket_intr_vectors = []
        for i in range(pocket_n):
            if i + ligand_n in pipi:
                vec = self._get_one_hot_vector("pipi", INTERACTION_TYPES)
            elif i + ligand_n in anion:
                vec = self._get_one_hot_vector("anion", INTERACTION_TYPES)
            elif i + ligand_n in cation:
                vec = self._get_one_hot_vector("cation", INTERACTION_TYPES)
            elif i + ligand_n in hbd:
                vec = self._get_one_hot_vector("hbd", INTERACTION_TYPES)
            elif i + ligand_n in hba:
                vec = self._get_one_hot_vector("hba", INTERACTION_TYPES)
            elif i + ligand_n in hydro:
                vec = self._get_one_hot_vector("hydro", INTERACTION_TYPES)
            else:
                vec = self._get_one_hot_vector("none", INTERACTION_TYPES)
            pocket_intr_vectors.append(vec)
        pocket_intr_mat = np.stack(pocket_intr_vectors, axis=0)
        if mask is not None:
            pocket_intr_mat = pocket_intr_mat * mask.reshape(-1, 1)
        return pocket_intr_mat


    # pocket-Ligand interaction (PLI) processor
    def pli_processor(self, ligand_fn, pocket_fn):
        try:
            ligand_mol = self.rd_file(ligand_fn) 
            assert ligand_mol is not None, "ligand_mol is None"

            pocket_mol, pocket_str, pocket_fn_2 = self._extract_binding_pocket(
                ligand_mol, pocket_fn, cutoff=10 ,use_whole_protein=False
            )            
            assert (
                pocket_str is not None
            )

            # pocket_mol = self.rd_file(pocket_fn)
            assert pocket_mol is not None, "pocket_mol is None"

            ligand_mol = Chem.RemoveHs(ligand_mol)
            pocket_mol = Chem.RemoveHs(pocket_mol)
            assert self._filter_ligand(ligand_mol), "ligand_mol is not valid"
            assert self._filter_pocket(pocket_mol), "pocket_mol is not valid"
       
            complex_mol, complex_fn, fd = self._join_complex(ligand_fn, pocket_fn_2)   
            assert complex_mol is not None, "complex_mol is None"
        
        except Exception as e:
            print(traceback.format_exc())
            return   
        
        ligand_n, pocket_n, complex_n = (
            ligand_mol.GetNumAtoms(),
            pocket_mol.GetNumAtoms(),
            complex_mol.GetNumAtoms(),
        )  

        interaction_info = self._get_complex_interaction_info(complex_fn)
        os.close(fd)
        os.remove(complex_fn)

        pocket_PLI_cond = self._get_pocket_interaction_matrix(
            ligand_n, pocket_n, interaction_info
        )

        dir_name = os.path.dirname(pocket_fn)
        pocket_fn_3 = os.path.join(dir_name, 'pocket_fn_3.pdb')
        Chem.MolToPDBFile(pocket_mol, pocket_fn_3)

        return pocket_PLI_cond, pocket_fn_3
    



def get_pocket_PLI_cond(ligand_fn, pocket_fn):
    pocket_PLI_cond =PLIC().pli_processor(ligand_fn, pocket_fn)
    return pocket_PLI_cond


if __name__ == "__main__":
    import argparse

    def get_dataset(raw_path, split_data_path):
        raw_dataset = PocketLigandPairDataset(raw_path)
        split_data = torch.load(split_data_path)
        subsets = {k: Subset(raw_dataset, indices=v) for k, v in split_data.items()}
        return raw_dataset, subsets


    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str,default='/PlicDiff/data_processed/data_processed_crossdock/crossdocked_v1.3_rmsd1.0_pocket10')
    parser.add_argument('--split_data_path', type=str,default='DL_code/PlicDiff/data_processed/data_processed_crossdock/crossdocked_pocket10_pose_split.pt')
    parser.add_argument('--base_path', type=str,default='/DL_code/PlicDiff/data_processed/data_processed_crossdock/crossdocked_v1.3_rmsd1.0_pocket10')
    parser.add_argument('--num_workers', type=int, default=10)
    args = parser.parse_args()

    processor = PLIC()  
    dataset, subsets = get_dataset(args.raw_path, args.split_data_path)
    print(type(dataset))

    pli_data = {}
    for i, data in enumerate(tqdm(dataset, desc='extarcting interaction condition', bar_format='{l_bar}{bar:100}{r_bar}{bar:-20b}')):
        try:

            pocket_fn = os.path.join(args.base_path, data.protein_filename)  
            ligand_fn = os.path.join(args.base_path, data.ligand_filename)

            pocket_pli_cond = processor.pli_processor(ligand_fn, pocket_fn)

            data.pocket_interaction_condition = pocket_pli_cond
            if i == 0:
                print('pocket_pli_cond.shape:', pocket_pli_cond.shape)
                print('data.pocket_interaction_condition.shape:', data.pocket_interaction_condition.shape)
            pli_data[i] = data

        except Exception as e:
            print(f"Error processing data at index {i}: {e}")
        
    with open(os.path.join(os.path.dirname(args.raw_path), 'data_list.pkl'), 'wb') as f:
        pickle.dump(pli_data, f)