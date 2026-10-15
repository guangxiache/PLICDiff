from openbabel import pybel
from meeko import MoleculePreparation
import meeko
from vina import Vina
import subprocess
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import tempfile
import os
import contextlib
import sys
from pathlib import Path

from utils.reconstruct import reconstruct_from_generated
from utils.evaluation.docking_qvina import get_random_id, BaseDockingTask
from utils.evaluation.executables import receptor_preparation_command


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf': 
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')
        
    def addH(self, polaronly=False, correctforph=True, PH=7): 
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)

    def gen_conf(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        if hasattr(meeko, 'PDBQTWriterLegacy'):
            rdmol = Chem.MolFromMolBlock(self.ob_mol.write('sdf'), removeHs=False)
            if rdmol is None:
                raise ValueError('Cannot parse ligand for Meeko preparation.')
            rdmol = Chem.AddHs(rdmol, addCoords=True)
            setups = preparator.prepare(rdmol)
            pdbqt, ok, error = meeko.PDBQTWriterLegacy.write_string(setups[0])
            if not ok:
                raise ValueError(error)
        else:
            preparator.prepare(self.ob_mol.OBMol)
            pdbqt = preparator.write_pdbqt_string()
        if lig_pdbqt is not None: 
            Path(lig_pdbqt).write_text(pdbqt)
            return 
        else: 
            return pdbqt
        

class PrepProt(object): 
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def del_water(self, dry_pdb_file): # optional
        with open(self.prot) as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
            dry_lines = [l for l in lines if not 'HOH' in l]
        
        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file
        
    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        # subprocess.Popen(['pdb2pqr30','--ff=AMBER',self.prot, self.prot_pqr],
        #                  stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()
        subprocess.run([
                        sys.executable, '-m', 'pdb2pqr',
                        '--ff=AMBER',
                        self.prot,
                        self.prot_pqr
                        ], check=True, capture_output=True, text=True)


    def get_pdbqt(self, prot_pdbqt):
        subprocess.run(receptor_preparation_command() + ['-r', self.prot_pqr, '-o', prot_pdbqt],
                       check=True, capture_output=True, text=True)
        if not Path(prot_pdbqt).is_file() or not Path(prot_pdbqt).stat().st_size:
            raise RuntimeError('Receptor preparation did not produce a PDBQT file.')


class VinaDock(object): 
    def __init__(self, lig_pdbqt, prot_pdbqt): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')]
            xs = [float(l[30:38]) for l in lines]
            ys = [float(l[38:46]) for l in lines]
            zs = [float(l[46:54]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None: 
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, **kwargs):  # seed=0 mean random seed
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only': 
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
        else:
            raise ValueError
        
        if not save_pose: 
            return score
        else: 
            if mode == 'score_only': 
                pose = None 
            elif mode == 'minimize': 
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f: 
                    v.write_pose(tmp.name, overwrite=True)             
                with open(tmp.name, 'r') as f: 
                    pose = f.read()
   
            elif mode == 'dock': 
                pose = v.poses(n_poses=1)
            else:
                raise ValueError
            return score, pose


class VinaDockingTask(BaseDockingTask):


    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        # protein_fn = os.path.join(
        #     os.path.dirname(ligand_filename),
        #     os.path.basename(ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        # )
        # protein_path = os.path.join(protein_root, protein_fn)
        if Path(protein_root).is_dir():
            protein_fn = os.path.join(os.path.dirname(ligand_filename),
                                      os.path.basename(ligand_filename)[:10] + '.pdb')
            protein_path = os.path.join(protein_root, protein_fn)
        else:
            protein_path = protein_root

        return cls(protein_path, ligand_rdmol, **kwargs)

    def __init__(self, protein_path, ligand_rdmol, tmp_dir='./tmp', center=None,
                 size_factor=1., buffer=5.0):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        if not Path(protein_path).is_file():
            raise FileNotFoundError(f'Protein PDB/PDBQT not found: {protein_path}')
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        sdf_writer = Chem.SDWriter(self.ligand_path)  
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, mode='dock', exhaustiveness=8, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr = os.path.join(self.tmp_dir, self.receptor_id + '.pqr')
        protein_pdbqt = os.path.join(self.tmp_dir, self.receptor_id + '.pdbqt')

        lig = PrepLig(self.ligand_path, 'sdf')
        lig.get_pdbqt(ligand_pdbqt)

        if self.receptor_path.lower().endswith('.pdbqt'):
            protein_pdbqt = self.receptor_path
        else:
            prot = PrepProt(self.receptor_path)
            if not os.path.exists(protein_pqr):
                prot.addH(protein_pqr)
            if not os.path.exists(protein_pdbqt):
                prot.get_pdbqt(protein_pdbqt)

        dock = VinaDock(ligand_pdbqt, protein_pdbqt)
        dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        score, pose = dock.dock(score_func='vina', mode=mode, exhaustiveness=exhaustiveness, save_pose=True, **kwargs)
        return [{'affinity': score, 'pose': pose}]


# if __name__ == '__main__':
#     lig_pdbqt = 'data/lig.pdbqt'
#     mol_file = 'data/1a4k_ligand.sdf'
#     a = PrepLig(mol_file, 'sdf')
#     # mol_file = 'CC(=C)C(=O)OCCN(C)C'
#     # a = PrepLig(mol_file, 'smi')
#     a.addH()
#     a.gen_conf()
#     a.get_pdbqt(lig_pdbqt)
#
#     prot_file = 'data/1a4k_protein_chainAB.pdb'
#     prot_dry = 'data/protein_dry.pdb'
#     prot_pqr = 'data/protein.pqr'
#     prot_pdbqt = 'data/protein.pdbqt'
#     b = PrepProt(prot_file)
#     b.del_water(prot_dry)
#     b.addH(prot_pqr)
#     b.get_pdbqt(prot_pdbqt)
#
#     dock = VinaDock(lig_pdbqt, prot_pdbqt)
#     dock.get_box()
#     dock.dock()
    
