"""Compare pocket means using paired pocket bootstrap, rather than molecule bootstrap."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np


def compare_pockets(paths, reference_model='PLICDiff', metric='pli_similarity',
                    n_bootstrap=20000, seed=2024):
    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be positive')
    grouped, seen = defaultdict(lambda: defaultdict(list)), set()
    models = set()
    if len({Path(path).resolve() for path in paths}) != len(paths):
        raise ValueError('Duplicate CSV input')
    for path in paths:
        with Path(path).open(newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'model', 'pocket_id', 'sample_id', metric}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{path} must have columns {sorted(required)}')
            for row in reader:
                model, pocket = row['model'], row['pocket_id']
                if not model or not pocket:
                    raise ValueError('Every row requires explicit model and pocket_id labels')
                models.add(model)
                grouped[model][pocket]  # Retain pockets where all metric calculations failed.
                key = (model, pocket, row['sample_id'])
                if key in seen:
                    raise ValueError(f'Duplicate sample: {key}')
                seen.add(key)
                if row[metric] == '':
                    continue
                value = float(row[metric])
                if not np.isfinite(value):
                    raise ValueError(f'Nonfinite {metric}: {key}')
                grouped[model][pocket].append(value)
    if reference_model not in models or len(models) < 2:
        raise ValueError('Need the reference model and at least one baseline')
    pockets = sorted(set().union(*(set(grouped[model]) for model in models)))
    if len(pockets) < 2:
        raise ValueError('At least two paired pockets are needed')
    means = {}
    pocket_rows = []
    for model in sorted(models):
        values = []
        for pocket in pockets:
            scores = grouped[model].get(pocket, [])
            if not scores:
                raise ValueError(f'No successful {metric} values for {model}/{pocket}; do not silently drop pockets')
            mean = float(np.mean(scores))
            values.append(mean)
            pocket_rows.append(dict(model=model, pocket_id=pocket, n=len(scores), mean=mean))
        means[model] = np.asarray(values)
    draws = np.random.default_rng(seed).integers(0, len(pockets), size=(n_bootstrap, len(pockets)))
    comparisons, differences = [], []
    for model in sorted(models - {reference_model}):
        delta = means[reference_model] - means[model]
        low, high = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
        comparisons.append(dict(reference_model=reference_model, baseline=model, metric=metric,
                                n_pockets=len(pockets), mean_difference=float(delta.mean()),
                                ci95_low=float(low), ci95_high=float(high),
                                n_positive=int((delta > 0).sum()), n_negative=int((delta < 0).sum()),
                                n_tied=int((delta == 0).sum())))
        differences.extend(dict(baseline=model, pocket_id=pocket, difference=float(value))
                           for pocket, value in zip(pockets, delta))
    return dict(metric=metric, reference_model=reference_model, pockets=pockets,
                difference_definition='reference model mean minus baseline mean within each pocket',
                weighting='equal weight per pocket', n_bootstrap=n_bootstrap, seed=seed,
                interval='95% percentile paired pocket bootstrap',
                comparisons=comparisons), pocket_rows, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='molecules.csv files from evaluate_molecules/evaluate_diffusion.')
    parser.add_argument('--reference_model', default='PLICDiff')
    parser.add_argument('--metric', default='pli_similarity')
    parser.add_argument('--n_bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    try:
        summary, pocket_rows, differences = compare_pockets(
            args.inputs, args.reference_model, args.metric, args.n_bootstrap, args.seed)
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    for filename, rows in [('pocket_means.csv', pocket_rows), ('paired_differences.csv', differences)]:
        with (output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
