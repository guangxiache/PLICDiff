"""Evaluate SDF records from any model, one protein pocket per invocation."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rdkit import Chem, RDLogger
from utils.evaluation.cli import add_evaluation_arguments, make_evaluator, validate_evaluation_arguments


def input_files(parser, args):
    if sum((bool(args.inputs), args.sdf_dir is not None, args.manifest is not None)) != 1:
        parser.error('Provide SDF paths, --sdf_dir, or --manifest (exactly one source)')
    if args.manifest:
        manifest = Path(args.manifest)
        with manifest.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if 'path' not in (reader.fieldnames or []):
                parser.error('Manifest CSV requires a path column')
            files = []
            for idx, row in enumerate(reader):
                if not row['path'].strip():
                    parser.error(f'Manifest row {idx + 2} has an empty path')
                path = Path(row['path']).expanduser()
                path = path if path.is_absolute() else manifest.parent / path
                files.append((path, row.get('sample_id') or f'{idx:05d}:{path.name}'))
    else:
        paths = sorted(Path(args.sdf_dir).glob('*.sdf')) if args.sdf_dir else map(Path, args.inputs)
        files = [(path, f'{idx:05d}:{path.name}') for idx, path in enumerate(paths)]
    if not files:
        parser.error('No input SDF files')
    if len({path.resolve() for path, _ in files}) != len(files):
        parser.error('Duplicate input files would count molecules twice')
    if len({sample_id for _, sample_id in files}) != len(files):
        parser.error('Manifest sample_id values must be unique')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='*', help='Explicit SDF paths; multi-record files are supported.')
    parser.add_argument('--sdf_dir', help='Read *.sdf directly in this directory; no recursion or GS inference.')
    parser.add_argument('--manifest', help='CSV with path and optional sample_id; relative paths use its directory.')
    parser.add_argument('--num_attempts', type=int, help='Total generated attempts, including missing/failed outputs.')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--verbose', action='store_true')
    add_evaluation_arguments(parser)
    args = parser.parse_args()
    validate_evaluation_arguments(parser, args)
    if args.num_attempts is not None and args.num_attempts < 1:
        parser.error('--num_attempts must be positive')
    files = input_files(parser, args)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')
    evaluator = make_evaluator(args, args.out_dir)
    for path, sample_id in files:
        if not path.is_file():
            evaluator.add(None, sample_id + '#0', source=path, input_status='input_failed',
                          input_error='Input file does not exist')
            continue
        record_idx = 0
        try:
            supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
            for record_idx, mol in enumerate(supplier, start=1):
                evaluator.add(mol, f'{sample_id}#{record_idx - 1}', source=path,
                              record_index=record_idx - 1, input_error='SDF record parsing failed')
            if record_idx == 0:
                raise ValueError('Empty SDF file')
        except Exception as error:
            evaluator.add(None, f'{sample_id}#{record_idx}', source=path, record_index=record_idx,
                          input_status='input_failed', input_error=str(error))
    try:
        summary = evaluator.write(args.out_dir, args.num_attempts)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({'n_input_records': summary['n_input_records'], 'n_valid': summary['n_valid'],
                      'validity': summary['validity'], 'uniqueness': summary['uniqueness'],
                      'out_dir': str(Path(args.out_dir).resolve())}, allow_nan=False))


if __name__ == '__main__':
    main()
