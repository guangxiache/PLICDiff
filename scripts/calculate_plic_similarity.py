"""Compare generated ligands on one fixed reference-defined protein pocket."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import sys
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from tqdm.auto import tqdm
from datasets.extract_PLIC import PLIC


def cosine_similarity(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f'PLI atom alignment mismatch: {x.shape} versus {y.shape}')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite PLI features')
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(np.dot(x, y) / norm, -1.0, 1.0)) if norm else 0.0


def group_sdf_files_by_guidance_scale(ligand_dir):
    groups = defaultdict(list)
    for path in sorted(Path(ligand_dir).glob('*.sdf')):
        match = re.search(r'_(pos[-+\deE.]+_v[-+\deE.]+|[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)$', path.stem)
        groups[match.group(1) if match else 'all'].append(path)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference_ligand', '--row_ligand_fn', dest='reference_ligand', required=True)
    parser.add_argument('--ligand_dir', required=True)
    parser.add_argument('--protein_fn', required=True)
    parser.add_argument('--radius', type=float, default=10.0)
    parser.add_argument('--out_dir')
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()
    if args.radius <= 0 or args.top_k < 0:
        parser.error('--radius must be positive and --top_k nonnegative')
    groups = group_sdf_files_by_guidance_scale(args.ligand_dir)
    if not groups:
        parser.error('No SDF files found in --ligand_dir')
    out = Path(args.out_dir or str(Path(args.ligand_dir).with_name(Path(args.ligand_dir).name + '_similarity_output')))
    out.mkdir(parents=True, exist_ok=True)
    plic = PLIC()
    reference, pocket_path = plic.pli_processor(
        args.reference_ligand, args.protein_fn, already_cropped=False,
        return_pocket=True, cutoff=args.radius, pocket_output=out / 'reference_pocket.pdb',
    )
    if reference is None:
        raise RuntimeError('Reference PLI extraction failed; no scores were assigned.')
    reference = reference[:, :6]
    if not np.any(reference):
        raise ValueError('Reference has no detected interactions; PLI cosine similarity is not informative.')
    rows, summary = [], {}
    for group, paths in groups.items():
        scores, failures = {}, []
        for path in tqdm(paths, desc=f'PLI {group}'):
            try:
                # Every ligand uses the same pocket and atom order. Never crop per generated ligand.
                profile = plic.pli_processor(str(path), pocket_path, already_cropped=True)
                if profile is None:
                    raise ValueError('PLIP/ligand parsing failed')
                score = cosine_similarity(reference, profile[:, :6])
                if not math.isfinite(score):
                    raise ValueError('Nonfinite similarity')
                scores[path.stem] = score
                rows.append({'group': group, 'filename': path.name, 'status': 'ok',
                             'pli_similarity': score, 'error': ''})
            except Exception as error:
                failures.append(path.name)
                rows.append({'group': group, 'filename': path.name, 'status': 'failed',
                             'pli_similarity': '', 'error': str(error)})
        ranked = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        mean = float(np.mean(list(scores.values()))) if scores else None
        summary[group] = {'n_input': len(paths), 'n_success': len(scores), 'n_failed': len(failures),
                          'mean_pli_similarity': mean, 'failed_files': failures}
        torch.save({**{f'{key}_similarity': val for key, val in ranked.items()},
                    'average_similarity': mean, 'successful_calculations_count': len(scores),
                    'failed_files': failures}, out / f'similarity_gs_{group}.pt')
        if args.top_k and ranked:
            top_dir = out / f'top_{args.top_k}_similarity_gs_{group}'
            top_dir.mkdir(exist_ok=True)
            for stem in list(ranked)[:args.top_k]:
                shutil.copy2(Path(args.ligand_dir) / f'{stem}.sdf', top_dir / f'{stem}.sdf')
    with (out / 'pli_similarity.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['group', 'filename', 'status', 'pli_similarity', 'error'])
        writer.writeheader()
        writer.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
