#!/usr/bin/env python3
"""Export pocket.pdb and its aligned seven-channel pli_profile.npy."""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.extract_PLIC import PLIC

INTERACTION_TYPES = ['pipi', 'anion', 'cation', 'hbd', 'hba', 'hydro', 'none']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ligand', type=str, required=True, help='Reference ligand SDF (used only to extract the profile)')
    parser.add_argument('--protein', type=str, required=True, help='Full protein PDB (the pocket is re-cropped inside by cutoff)')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--cutoff', type=float, default=10.0)
    args = parser.parse_args()

    for fn in (args.ligand, args.protein):
        if not os.path.exists(fn):
            sys.exit(f'File not found: {fn}')

    os.makedirs(args.out_dir, exist_ok=True)
    pocket_out = os.path.join(args.out_dir, 'pocket.pdb')
    plic = PLIC()
    pli, _ = plic.pli_processor(
        args.ligand, args.protein, already_cropped=False, return_pocket=True,
        cutoff=args.cutoff, pocket_output=pocket_out)
    if pli is None:
        sys.exit('Profile extraction failed (pli_processor returned None); check the ligand and protein files')

    npy_path = os.path.join(args.out_dir, 'pli_profile.npy')
    np.save(npy_path, np.asarray(pli, dtype=np.float32))

    pli = np.asarray(pli)
    counts = {t: int(pli[:, i].sum()) for i, t in enumerate(INTERACTION_TYPES)}
    print(f'profile: {npy_path}  shape={pli.shape}')
    print(f'pocket:  {pocket_out}')
    print('Interaction counts: ' + ', '.join(f'{k}={v}' for k, v in counts.items()))
    print('Sampling example:')
    print(f'  python scripts/sample_for_pocket.py configs/sampling.yml '
          f'--pdb_path {pocket_out} --pli_path {npy_path} --result_path <out>')


if __name__ == '__main__':
    main()
