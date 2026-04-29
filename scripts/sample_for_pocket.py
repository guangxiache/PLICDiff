import sys
sys.path.append('/home/swhuang/DL_code/PlicDiff_3_clip')
import argparse
import os
import shutil

import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model_pli_cross_atn import ScorePosNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
from utils.data import PDBProtein, parse_sdf_file
from utils import reconstruct
from rdkit import Chem
from extract_PLIC import PLIC 


def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )
    return data

def raw_ligand_data(raw_ligand_path):
    raw_ligand_dict = parse_sdf_file(raw_ligand_path)
    data = ProteinLigandData.from_protein_ligand_dicts(
        ligand_dict=torchify_dict(raw_ligand_dict),
        )
    return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--pdb_path', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--result_path', type=str, default='./7ama_DC_853_gen_gs_0point3')
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--reference_ligand', action='store_true')
    parser.add_argument('--raw_ligand_path', type=str)
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    if args.reference_ligand:
        transform_raw_ligand = Compose([
            ligand_featurizer,
        ])

    transform = Compose([
        protein_featurizer, 
    ])
    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # extract pli as condition
    plic = PLIC()
    if args.reference_ligand:
        print('extract pli as condition')
        pli, pocket_path = plic.pli_processor(args.raw_ligand_path, args.pdb_path)
        pli = torch.from_numpy(pli).to(torch.float32) 
    else:
        pli = None


    # Load pocket
    data = pdb_to_pocket_data(pocket_path)
    data = transform(data)

    # load raw ligand if reference_ligand is True
    if args.reference_ligand:
        data_raw_ligand = raw_ligand_data(args.raw_ligand_path)
        data_raw_ligand = transform_raw_ligand(data_raw_ligand)
    else:
        data_raw_ligand = None

    if args.num_samples:
        config.sample.num_samples = args.num_samples

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    mols_save_path = os.path.join(result_path, f'sdf')
    
    # for guidance_scale in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for guidance_scale in [0.3]:


        guidance_scale_pos=guidance_scale
        guidence_scale_v=guidance_scale
        all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data, data_raw_ligand, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            pli=pli,
            guidance_scale_pos=guidance_scale_pos,
            guidance_scale_v=guidence_scale_v,        
        )
        result = {
            'data': data,
            'pred_ligand_pos': all_pred_pos,
            'pred_ligand_v': all_pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj
        }
        logger.info(f'Sample done! Guidence scale is {guidance_scale}')

        # reconstruction
        gen_mols = []
        n_recon_success, n_complete = 0, 0
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
            pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode='add_aromatic')
            try:
                pred_aromatic = trans.is_aromatic_from_index(pred_v, mode='add_aromatic')
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                gen_mols.append(None)
                continue
            n_recon_success += 1

            if '.' in smiles:
                gen_mols.append(None)
                continue
            n_complete += 1
            gen_mols.append(mol)
        result['mols'] = gen_mols
        logger.info('Reconstruction done!')
        logger.info(f'n recon: {n_recon_success} n complete: {n_complete}')


        torch.save(result, os.path.join(result_path, f'results_{guidance_scale}.pt'))

        os.makedirs(mols_save_path, exist_ok=True)
        for idx, mol in enumerate(gen_mols):
            if mol is not None:
                sdf_writer = Chem.SDWriter(os.path.join(mols_save_path, f'{idx:03d}'+'_'+f'{guidance_scale}.sdf'))
                sdf_writer.write(mol)
                sdf_writer.close()
    logger.info(f'Results are saved in {result_path}')
