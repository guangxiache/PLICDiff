"""Evaluate PLICDiff PT sampling outputs with shared, model-independent metrics."""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use('Agg')
import numpy as np
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm

from utils import misc, reconstruct, transforms
from utils.runtime import parse_bool
from utils.evaluation import eval_bond_length
from utils.evaluation.cli import add_evaluation_arguments, make_evaluator, validate_evaluation_arguments


def extract_idx_from_path(path):
    return Path(path).stem.removeprefix('results_').removeprefix('result_')


def sample_arrays(result, eval_step):
    """Accept stored trajectories or final-only sampling outputs without silent truncation."""
    pos = result.get('pred_ligand_pos_traj')
    types = result.get('pred_ligand_v_traj')
    if pos is not None and len(pos):
        if types is None or len(pos) != len(types):
            raise ValueError('Position/type trajectory sample counts differ')
        return [(np.asarray(p[eval_step]), np.asarray(v[eval_step])) for p, v in zip(pos, types)]
    if eval_step != -1:
        raise ValueError('Final-only output supports only --eval_step -1')
    pos, types = result['pred_ligand_pos'], result['pred_ligand_v']
    if len(pos) != len(types):
        raise ValueError('Final position/type sample counts differ')
    return [(np.asarray(p), np.asarray(v)) for p, v in zip(pos, types)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample_path')
    parser.add_argument('--verbose', type=parse_bool, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int)
    parser.add_argument('--save', type=parse_bool, default=True)
    parser.add_argument('--out_dir')
    parser.add_argument('--atom_enc_mode', default='add_aromatic', choices=['basic', 'add_aromatic', 'full'])
    parser.add_argument('--sdf_path', default='', help='Legacy ligand_filename prefix in metrics PT output.')
    add_evaluation_arguments(parser, allow_qvina=True)
    args = parser.parse_args()
    validate_evaluation_arguments(parser, args)
    if args.eval_num_examples is not None and args.eval_num_examples < 1:
        parser.error('--eval_num_examples must be positive')
    files = sorted(set(Path(args.sample_path).glob('results_*.pt')) |
                   set(Path(args.sample_path).glob('result_*.pt')))
    if args.eval_num_examples is not None:
        files = files[:args.eval_num_examples]
    if not files:
        parser.error('No results_*.pt or result_*.pt sampling outputs found')
    if len({extract_idx_from_path(path) for path in files}) != len(files):
        parser.error('result_ and results_ files have colliding output identifiers; evaluate them separately')
    output = Path(args.out_dir or Path(args.sample_path) / 'eval_results')
    output.mkdir(parents=True, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=str(output))
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')
    summaries = []
    for path in tqdm(files, desc='Evaluate'):
        result = torch.load(path, map_location='cpu', weights_only=False)
        arrays = sample_arrays(result, args.eval_step)
        evaluator = make_evaluator(args, output, coordinates_source='raw generated coordinates before reconstruction')
        evaluator.protocol.update(eval_step=args.eval_step, atom_enc_mode=args.atom_enc_mode,
                                  sampling_metadata=result.get('sampling_metadata', {}))
        idx = extract_idx_from_path(path)
        legacy_results = []
        for sample_idx, (positions, atom_types) in enumerate(arrays):
            mol, atom_numbers, error = None, None, ''
            try:
                if positions.ndim != 2 or positions.shape != (len(atom_types), 3):
                    raise ValueError('Coordinate/type shape mismatch')
                if not len(atom_types) or not np.isfinite(positions).all():
                    raise ValueError('Empty sample or nonfinite coordinates')
                atom_numbers = transforms.get_atomic_number_from_index(atom_types, mode=args.atom_enc_mode)
                aromatic = transforms.is_aromatic_from_index(atom_types, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(positions, atom_numbers, aromatic)
            except Exception as exception:
                error = str(exception)
            reconstructed = mol is not None
            data = result.get('data')
            docking_filename = getattr(data, 'ligand_filename', None)
            clean, row, chem, vina = evaluator.add(
                mol, sample_id=f'{idx}:{sample_idx}', source=str(path), record_index=sample_idx,
                positions=positions, atom_numbers=atom_numbers, reconstructed=reconstructed,
                input_status='reconstruction_failed', input_error=error,
                docking_filename=docking_filename)
            if row['chemistry_status'] == 'ok':
                # Preserve the legacy PT schema, including raw predictions and RDKit molecules.
                legacy_results.append(dict(
                    mol=clean, smiles=row['smiles'],
                    ligand_filename=str(Path(args.sdf_path) / f'{sample_idx:03d}_{idx}.sdf'),
                    pred_pos=positions, pred_v=atom_types, chem_results=chem, vina=vina,
                    pli_similarity=row['pli_similarity']))
            if args.verbose and row['status'] != 'valid':
                logger.warning('%s: %s', row['sample_id'], row['error'])
        summary = evaluator.summary(num_attempts=len(arrays) if arrays else None)
        summaries.append({'input': str(path), 'summary': summary})
        logger.info('%s: attempts=%d valid=%d unique=%d', path.name, len(arrays),
                    summary['n_valid'], summary['n_unique'])
        for name in ('qed', 'sa', 'pli_similarity', 'vina_score', 'vina_min'):
            stat = summary['metrics'][name]
            logger.info('%s: n=%d mean=%s', name, stat['n'], stat['mean'])
        if args.save:
            prefix = f'{args.eval_step}_{idx}_'
            evaluator.write(output, num_attempts=len(arrays) if arrays else None, prefix=prefix)
            stability = summary['stability']
            n = len(arrays)
            legacy_validity = dict(
                mol_stable=stability['molecule_stability'], atm_stable=stability['atom_stability'],
                recon_success=summary['reconstruction_success'],
                complete=summary['validity'],
                eval_success=len(legacy_results) / n if n else None)
            torch.save(dict(stability=legacy_validity, bond_length=evaluator.bond_distances,
                            all_results=legacy_results, summary=summary),
                       output / f'metrics_{args.eval_step}_{idx}.pt')
            pair_profile = eval_bond_length.get_pair_length_profile(evaluator.pair_distances)
            eval_bond_length.plot_distance_hist(
                pair_profile, summary['geometry']['pair_distance_jsd'],
                str(output / f'pair_dist_hist_{args.eval_step}_{idx}.png'))
    if args.save:
        (output / f'summaries_{args.eval_step}.json').write_text(
            json.dumps(summaries, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
