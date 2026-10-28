import sys
import argparse
import os
import shutil
import math
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
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
from datasets.extract_PLIC import PLIC
from utils.runtime import load_checkpoint, project_path, validate_device


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


def save_molecules(molecules, output_dir, guidance_scale):
    """Replace the SDF outputs for one guidance setting."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f'_{guidance_scale}'
    for path in directory.glob(f'*{suffix}.sdf'):
        if path.stem.removesuffix(suffix).isdigit():
            path.unlink()
    for idx, mol in enumerate(molecules):
        if mol is not None:
            with Chem.SDWriter(str(directory / f'{idx:03d}{suffix}.sdf')) as writer:
                writer.write(mol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--pdb_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--result_path', type=str, default='./outputs/sample_pocket')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--guidance_scale', type=float, nargs='+',
                        help='Shared null-origin CFG coefficient(s); overrides the configuration.')
    parser.add_argument('--guidance_scale_pos', type=float)
    parser.add_argument('--guidance_scale_v', type=float)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--reference_ligand', action='store_true')
    parser.add_argument('--raw_ligand_path', type=str)
    parser.add_argument('--pli_path', type=str, help='de novo mode: load a PLI profile (.npy, shape=(n_pocket_atoms, 7)) instead of a reference ligand')
    args = parser.parse_args()
    args.device = validate_device(args.device)
    args.config = project_path(args.config)
    if not Path(args.pdb_path).is_file():
        parser.error('--pdb_path must name a protein/pocket PDB file')
    if args.reference_ligand and (not args.raw_ligand_path or not Path(args.raw_ligand_path).is_file()):
        parser.error('--reference_ligand requires an existing --raw_ligand_path')
    if args.guidance_scale is not None and (args.guidance_scale_pos is not None or args.guidance_scale_v is not None):
        parser.error('Use --guidance_scale or separate coordinate/type coefficients, not both.')
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error('--num_samples must be positive')
    if args.batch_size <= 0:
        parser.error('--batch_size must be positive')

    logger = misc.get_logger('evaluate')

    config = misc.load_config(args.config)
    if args.checkpoint:
        config.model.checkpoint = project_path(args.checkpoint)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    ckpt = load_checkpoint(args.checkpoint or config.model.checkpoint, args.device)
    logger.info(f"Training Config: {ckpt['config']}")

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
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    model.eval()
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    plic = PLIC()
    if args.reference_ligand and args.pli_path:
        raise ValueError('--reference_ligand and --pli_path are mutually exclusive')
    if args.reference_ligand:
        print('extract pli as condition')
        pli, pocket_path = plic.pli_processor(
            args.raw_ligand_path, args.pdb_path,
            already_cropped=False, return_pocket=True,
            pocket_output=os.path.join(args.result_path, 'pocket.pdb'),
        )
        if pli is None:
            raise RuntimeError('PLI extraction failed; inspect the reported protein/ligand error.')
        pli = torch.from_numpy(pli).to(torch.float32)
    elif args.pli_path:
        pli = torch.from_numpy(np.load(args.pli_path)).to(torch.float32)
        pocket_path = args.pdb_path
        print(f'load pli profile from {args.pli_path}, shape={tuple(pli.shape)}')
    else:
        pli = None
        pocket_path = args.pdb_path


    data = pdb_to_pocket_data(pocket_path)
    data = transform(data)
    if pli is not None and (pli.ndim != 2 or pli.shape != (len(data.protein_pos), 7)):
        raise ValueError(
            f'PLI shape {tuple(pli.shape)} does not match ({len(data.protein_pos)}, 7): '
            'The profile must match the pocket given by --pdb_path atom by atom. '
            'Use the pocket exported together with the profile, or re-extract it.')
    if pli is not None and (not torch.isfinite(pli).all() or (pli < 0).any()):
        raise ValueError('PLI profile must contain finite, nonnegative values.')
    if pli is not None:
        pli = pli.to(args.device)

    if args.reference_ligand:
        data_raw_ligand = raw_ligand_data(args.raw_ligand_path)
        data_raw_ligand = transform_raw_ligand(data_raw_ligand)
    else:
        data_raw_ligand = None

    if args.num_samples:
        config.sample.num_samples = args.num_samples
    if args.num_steps is not None:
        config.sample.num_steps = args.num_steps
    if config.sample.pos_only:
        config.sample.sample_num_atoms = 'ref'
    default_pos = float(config.sample.get('guidance_scale_pos', 1.3))
    default_v = float(config.sample.get('guidance_scale_v', 1.3))
    if args.guidance_scale is not None:
        guidance_pairs = [(x, x) for x in args.guidance_scale]
    else:
        guidance_pairs = [(args.guidance_scale_pos if args.guidance_scale_pos is not None else default_pos,
                           args.guidance_scale_v if args.guidance_scale_v is not None else default_v)]
    if any(not math.isfinite(x) for pair in guidance_pairs for x in pair):
        parser.error('Guidance coefficients must be finite')
    use_reference = bool(config.sample.get('guidance_use_reference_ligand', True))
    if data_raw_ligand is None:
        use_reference = False

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    mols_save_path = os.path.join(result_path, 'sdf')

    for guidance_scale_pos, guidance_scale_v in guidance_pairs:
        guidance_scale = (str(guidance_scale_pos) if guidance_scale_pos == guidance_scale_v
                          else f'pos{guidance_scale_pos}_v{guidance_scale_v}')
        all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data, data_raw_ligand, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            pli=pli,
            guidance_scale_pos=guidance_scale_pos,
            guidance_scale_v=guidance_scale_v,
            use_reference_ligand_in_guidance=use_reference,
        )
        result = {
            'data': data,
            'pred_ligand_pos': all_pred_pos,
            'pred_ligand_v': all_pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'sampling_metadata': {
                'guidance_convention': 'null + g * (conditional - null)',
                'guidance_scale_pos': guidance_scale_pos,
                'guidance_scale_v': guidance_scale_v,
                'use_reference_ligand_in_guidance': use_reference,
                'checkpoint': config.model.checkpoint,
                'num_samples': config.sample.num_samples,
                'batch_size': args.batch_size,
                'sample_num_atoms': config.sample.sample_num_atoms,
                'pos_only': config.sample.pos_only,
                'protein_path': str(Path(pocket_path).resolve()),
                'num_steps': config.sample.num_steps,
                'seed': config.sample.seed,
            },
        }
        logger.info(f'Sample done! Guidance scale is {guidance_scale} '
                    f'(use_reference_ligand_in_guidance={use_reference})')

        gen_mols = []
        n_recon_success, n_complete = 0, 0
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
            pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=ligand_atom_mode)
            try:
                pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=ligand_atom_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except Exception as error:
                logger.debug('Reconstruction failed for sample %d: %s', sample_idx, error)
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
        with open(os.path.join(result_path, f'sampling_{guidance_scale}.json'), 'w') as f:
            json.dump(result['sampling_metadata'], f, indent=2)

        save_molecules(gen_mols, mols_save_path, guidance_scale)
    logger.info(f'Results are saved in {result_path}')
