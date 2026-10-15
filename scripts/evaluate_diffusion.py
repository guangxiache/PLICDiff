import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import os

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
import matplotlib

# Evaluation writes image files and must not wait for a display/GUI connection.
matplotlib.use('Agg')

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.runtime import parse_bool


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        if len(all_ring_sizes) == 0:
            logger.info(f'ring size: {ring_size} ratio: None')
        else:
            logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


def extract_idx_from_path(path):
    filename = os.path.basename(path)
    idx = filename.split('_')[-1].split('.pt')[0]
    return filename[:-3].removeprefix('results_').removeprefix('result_')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--verbose', type=parse_bool, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=parse_bool, default=True)
    parser.add_argument('--protein_root', type=str, default=None,
                        help='Protein PDB for Vina, or protein dataset root for QVina.')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, default='none', choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--docking_cpu', type=int, default=1)
    parser.add_argument('--sdf_path',type=str, default='')
    args = parser.parse_args()
    if args.docking_cpu < 1:
        parser.error('--docking_cpu must be positive')
    if args.docking_mode == 'qvina':
        from utils.evaluation.docking_qvina import QVinaDockingTask
    elif args.docking_mode in {'vina_score', 'vina_dock'}:
        from utils.evaluation.docking_vina import VinaDockingTask
    if args.docking_mode != 'none' and not args.protein_root:
        parser.error('Docking requires --protein_root (protein PDB for Vina, dataset root for QVina).')

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = sorted(set(glob(os.path.join(args.sample_path, 'results_*.pt')) +
                                 glob(os.path.join(args.sample_path, 'result_*.pt'))))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    if not num_examples:
        raise FileNotFoundError('No results_*.pt or result_*.pt sampling outputs found.')
    logger.info(f'Load generated data done! {num_examples} examples in total.')


    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        num_samples = 0
        all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
        n_recon_success, n_eval_success, n_complete = 0, 0, 0
        results = []
        all_pair_dist, all_bond_dist = [], []
        all_atom_types = Counter()
        success_pair_dist, success_atom_types = [], Counter()

        r = torch.load(r_name, map_location='cpu', weights_only=False)
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)
        if not num_samples:
            logger.warning('Skipping empty sampling output: %s', r_name)
            continue

        idx = extract_idx_from_path(r_name)
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            all_atom_types += Counter(pred_atom_type)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            # reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue
            n_recon_success += 1

            if '.' in smiles:
                continue
            n_complete += 1

            # chemical and docking check
            try:
                chem_results = scoring_func.get_chem(mol)
                ligand_filename = os.path.join(args.sdf_path, f'{sample_idx:03d}_{idx}.sdf')
                if args.docking_mode == 'qvina':
                    vina_task = QVinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, protein_root=args.protein_root)
                    vina_results = vina_task.run_sync()
                elif args.docking_mode in ['vina_score', 'vina_dock']:
                    # vina_task = VinaDockingTask.from_generated_mol(
                    #     mol, r['data'].ligand_filename, protein_root=args.protein_root)
                    vina_task = VinaDockingTask.from_generated_mol(
                        mol, ligand_filename, protein_root=args.protein_root)

                    score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness, cpu=args.docking_cpu)
                    minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness, cpu=args.docking_cpu)
                    vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                    if args.docking_mode == 'vina_dock':
                        docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness, cpu=args.docking_cpu)
                        vina_results['dock'] = docking_results
                else:
                    vina_results = None

                n_eval_success += 1
            except Exception as e:
                if args.verbose:
                    logger.warning('Evaluation failed for %s: %s', f'{example_idx}_{sample_idx}', str(e))
                continue

            # now we only consider complete molecules as success
            # Bond length distribution is counted here
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            results.append({
                'mol': mol,
                'smiles': smiles,
                'ligand_filename': ligand_filename,          # r['data'].ligand_filename,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'chem_results': chem_results,
                'vina': vina_results
            })
    
        logger.info(f'Evaluate done! {num_samples} samples in total.')

        fraction_mol_stable = all_mol_stable / num_samples
        fraction_atm_stable = all_atom_stable / all_n_atom if all_n_atom else 0.0
        fraction_recon = n_recon_success / num_samples
        fraction_eval = n_eval_success / num_samples
        fraction_complete = n_complete / num_samples
        validity_dict = {
            'mol_stable': fraction_mol_stable,
            'atm_stable': fraction_atm_stable,
            'recon_success': fraction_recon,
            'eval_success': fraction_eval,
            'complete': fraction_complete
        }
        print_dict(validity_dict, logger)

        # Print bond length data
        c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
        c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
        logger.info('JS bond distances of complete mols: ')
        print_dict(c_bond_length_dict, logger)

        success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
        success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
        print_dict(success_js_metrics, logger)

        if len(success_atom_types)==0:
            atom_type_js = None
            logger.info(f'Atom type JS: {atom_type_js}')
        else:
            atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
            logger.info('Atom type JS: %.4f' % atom_type_js)

        if args.save:
            eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                                metrics=success_js_metrics,
                                                save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}_{idx}.png'))

        logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
            n_recon_success, n_complete, len(results)))

        qed = [r['chem_results']['qed'] for r in results]
        sa = [r['chem_results']['sa'] for r in results]
        logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
        logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
        if args.docking_mode == 'qvina':
            vina = [r['vina'][0]['affinity'] for r in results]
            logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
        elif args.docking_mode in ['vina_dock', 'vina_score']:
            vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
            vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
            logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
            logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
            if args.docking_mode == 'vina_dock':
                vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
                logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

        # check ring distribution
        print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

        if args.save:
            torch.save({
                'stability': validity_dict,
                'bond_length': all_bond_dist,
                'all_results': results
            }, os.path.join(result_path, f'metrics_{args.eval_step}_{idx}.pt'))
