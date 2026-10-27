"""Shared PLIP extraction with a fixed protein atom order and seven PLI channels."""
from rdkit import Chem
import os
import shutil
import traceback
import tempfile
from pathlib import Path
import logging
import numpy as np
from plip.structure.preparation import PDBComplex

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBIO import Select
from scipy.spatial import distance_matrix
from utils.data import PDBProtein

INTERACTION_TYPES = ["pipi", "anion", "cation", "hbd", "hba", "hydro"]


class PLIC:
    def __init__(
        self,
        seed=2024,
        tmp_dir=None
    ):

        self.seed = seed
        if tmp_dir is not None:
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        self._workspace = tempfile.TemporaryDirectory(prefix='plicdiff_', dir=tmp_dir)
        self._tmp_dir = self._workspace.name


    @staticmethod
    def _obabel():
        """Locate the obabel executable, searching sys.prefix/bin and PATH."""
        import sys
        exe = shutil.which("obabel")
        if exe:
            return exe
        cand = os.path.join(sys.prefix, "bin", "obabel")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
        raise RuntimeError(
            "obabel not found: activate the environment containing openbabel "
            "(micromamba activate plicdiff) or add its bin directory to PATH"
        )

    def check_and_create_directory(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Directory '{directory_path}' already exists.")


    def rd_file(self, fn):
        fn = str(fn)
        extension = Path(fn).suffix.lower().lstrip('.')
        if extension == "sdf":
            supplier = Chem.SDMolSupplier(fn)
            mol = supplier[0] if len(supplier) else None
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
        parser = PDBParser(QUIET=True)
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
                        if atom.element not in {"H", "D"}
                    ]
                )

                if not len(residue_positions):
                    return 0
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
        m2 = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
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
        """Write ligand first, then the exact model pocket order; never invoke a shell."""
        fd = None
        if complex_fn is None:
            fd, complex_fn = tempfile.mkstemp(
                suffix=".pdb", prefix="tmp_com_", dir=self._tmp_dir
            )
            os.close(fd)
            fd = None
        ligand = Chem.RemoveHs(self.rd_file(ligand_fn))
        pocket = PDBProtein(str(pocket_fn))
        ligand_block = Chem.MolToPDBBlock(ligand)
        atom_lines = []
        connections = []
        chains = {a['chain'] for a in pocket.atoms}
        chain = next((x for x in 'ZYXWVUTSRQPONMLKJIHGFEDCBA0123456789' if x not in chains), 'Z')
        for line in ligand_block.splitlines():
            if line.startswith(('ATOM  ', 'HETATM')):
                atom_lines.append('HETATM' + line[6:17] + 'LIG ' + chain + '   1' + line[26:])
            elif line.startswith('CONECT'):
                connections.append(line)
        for serial, atom in enumerate(pocket.atoms, len(atom_lines) + 1):
            atom_lines.append(f'ATOM  {serial:5d}' + atom['line'][11:])
        Path(complex_fn).write_text('\n'.join(atom_lines + connections + ['END', '']))
        complex_mol = Chem.MolFromPDBFile(complex_fn, sanitize=False, removeHs=False)
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
            raise ValueError('PLIP did not recognize the LIG residue in the assembled complex.')
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
    def pli_processor(self, ligand_fn, pocket_fn, already_cropped=True, return_pocket=False,
                      cutoff=10.0, pocket_output=None):
        """Compute the per-pocket-atom protein-ligand interaction condition matrix (7-channel one-hot).

        already_cropped=True  - training path: pocket_fn is an already cropped pocket file,
                                read directly.
        already_cropped=False - sampling path: pocket_fn is a full protein; the pocket is
                                re-cropped around the ligand with cutoff=10 A.
        return_pocket=True    - additionally return the path of the hydrogen-stripped pocket PDB.
        """
        try:
            ligand_mol = self.rd_file(ligand_fn)
            assert ligand_mol is not None, "ligand_mol is None"

            if not already_cropped:
                pocket_mol, pocket_str, pocket_fn_2 = self._extract_binding_pocket(
                    ligand_mol, pocket_fn, cutoff=cutoff, use_whole_protein=False
                )
                assert (
                    pocket_str is not None
                )
            ligand_mol = Chem.RemoveHs(ligand_mol)
            assert self._filter_ligand(ligand_mol), "ligand_mol is not valid"
            complex_pocket_fn = pocket_fn if already_cropped else pocket_fn_2
            protein = PDBProtein(str(complex_pocket_fn))
            if not protein.atoms:
                raise ValueError('No protein ATOM records in pocket.')
            if any(x == 1 for x in protein.element):
                if already_cropped and not return_pocket:
                    raise ValueError('Use a heavy-atom protein pocket; remove protein hydrogens before preprocessing.')
                heavy_lines = [a['line'] for a, e in zip(protein.atoms, protein.element) if e != 1]
                complex_pocket_fn = os.path.join(self._tmp_dir, 'heavy_pocket.pdb')
                Path(complex_pocket_fn).write_text('\n'.join(['HEADER    POCKET', *heavy_lines, 'END', '']))
                protein = PDBProtein(str(complex_pocket_fn))
            complex_mol, complex_fn, fd = self._join_complex(ligand_fn, complex_pocket_fn)
            assert complex_mol is not None, "complex_mol is None"
            if complex_mol.GetNumAtoms() != ligand_mol.GetNumAtoms() + len(protein.atoms):
                raise ValueError('Complex atom count changed; PLI atom alignment cannot be guaranteed.')
            interaction_info = self._get_complex_interaction_info(complex_fn)
            pocket_PLI_cond = self._get_pocket_interaction_matrix(
                ligand_mol.GetNumAtoms(), len(protein.atoms), interaction_info
            ).astype(np.float32)
            if return_pocket:
                if pocket_output is None:
                    pocket_output = os.path.join(self._tmp_dir, 'pocket.pdb')
                if Path(pocket_output).resolve() in {Path(pocket_fn).resolve(), Path(ligand_fn).resolve()}:
                    raise ValueError('pocket_output must differ from the input protein and ligand files.')
                Path(pocket_output).parent.mkdir(parents=True, exist_ok=True)
                Path(pocket_output).write_text('\n'.join(
                    ['HEADER    POCKET', *[atom['line'] for atom in protein.atoms], 'END', '']))
                return pocket_PLI_cond, str(pocket_output)
            return pocket_PLI_cond
        except Exception:
            logging.getLogger(__name__).warning('PLI extraction failed for %s:\n%s', ligand_fn, traceback.format_exc())
            return (None, None) if return_pocket else None
