"""Shared molecule metrics and explicit denominators for PT and SDF evaluation."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

from utils.evaluation import analyze, eval_atom_type, eval_bond_length, scoring_func
from utils.evaluation.similarity import pairwise_diversity, tanimoto_sim

SCALAR_METRICS = ('qed', 'sa', 'sa_raw', 'logp', 'lipinski', 'reference_tanimoto',
                  'pli_similarity', 'vina_score', 'vina_min', 'vina_dock', 'qvina')


def finite_summary(values):
    values = np.asarray([float(v) for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    return {'n': len(values), 'mean': float(values.mean()) if len(values) else None,
            'median': float(np.median(values)) if len(values) else None}


def clean_molecule(mol):
    """Validate a copy without generating coordinates or changing protonation."""
    if mol is None:
        raise ValueError('Molecule could not be parsed or reconstructed')
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    if not mol.GetNumHeavyAtoms():
        raise ValueError('No heavy atoms')
    return mol


def read_reference(path):
    if path is None:
        return None
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    if len(supplier) != 1:
        raise ValueError('Reference SDF must contain exactly one molecule')
    mol = clean_molecule(supplier[0])
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('Reference must be connected')
    return mol


class MoleculeMetrics:
    def __init__(self, model='', pocket_id='', reference=None, pli_scorer=None,
                 docking_scorer=None, diversity_radius=2, diversity_bits=2048,
                 diversity_chirality=False, coordinates_source='input SDF conformers'):
        self.model, self.pocket_id = model, pocket_id
        self.reference, self.pli_scorer, self.docking_scorer = reference, pli_scorer, docking_scorer
        self.fp_options = dict(kind='morgan', radius=diversity_radius, n_bits=diversity_bits,
                               use_chirality=diversity_chirality)
        self.coordinates_source = coordinates_source
        self.protocol = {}
        self.rows, self.molecules = [], []
        self.bond_distances, self.pair_distances = [], []
        self.atom_types = Counter()
        self.n_stable_mols = self.n_stability_mols = self.n_stable_atoms = self.n_atoms = 0

    def _coordinates(self, row, positions, atom_numbers):
        try:
            positions, atom_numbers = np.asarray(positions, dtype=float), np.asarray(atom_numbers, dtype=int)
            if not len(atom_numbers) or positions.shape != (len(atom_numbers), 3):
                raise ValueError('Coordinate/atom array shape mismatch or empty molecule')
            if not np.isfinite(positions).all() or (atom_numbers <= 0).any():
                raise ValueError('Nonfinite coordinates or invalid atomic numbers')
            pairs = eval_bond_length.pair_distance_from_pos_v(positions, atom_numbers)
            row['coordinate_status'] = 'ok'
        except Exception as error:
            row['coordinate_status'], row['coordinate_error'] = 'failed', str(error)
            return None, None
        try:
            unsupported = set(atom_numbers) - set(analyze.atom_decoder)
            if unsupported:
                raise ValueError(f'Stability thresholds unavailable for atomic numbers {sorted(unsupported)}')
            stable, n_stable, n_atoms = analyze.check_stability(positions, atom_numbers)
            row.update(stability_status='ok', molecule_stable=bool(stable),
                       n_stable_atoms=int(n_stable), n_atoms=int(n_atoms))
            self.n_stability_mols += 1
            self.n_stable_mols += int(stable)
            self.n_stable_atoms += int(n_stable)
            self.n_atoms += int(n_atoms)
        except Exception as error:
            row['stability_status'], row['stability_error'] = 'failed', str(error)
        return pairs, atom_numbers

    def add(self, mol, sample_id, source='', record_index=0, positions=None, atom_numbers=None,
            reconstructed=None, input_error='', input_status='invalid', docking_filename=None):
        row = dict(model=self.model, pocket_id=self.pocket_id, sample_id=str(sample_id), source=str(source),
                   record_index=record_index, status='invalid', smiles='', error='', reconstructed=reconstructed,
                   chemistry_status='not_applicable', chemistry_error='', coordinate_status='unavailable',
                   coordinate_error='', stability_status='unavailable', stability_error='',
                   molecule_stable=None, n_stable_atoms=None, n_atoms=None, bond_status='unavailable',
                   bond_error='', pli_status='not_requested' if self.pli_scorer is None else 'not_applicable',
                   pli_error='', docking_status='not_requested' if self.docking_scorer is None else 'not_applicable',
                   docking_error='', reference_status='not_requested' if self.reference is None else 'not_applicable',
                   reference_error='', **{key: None for key in SCALAR_METRICS})
        self.rows.append(row)
        pairs, elements = None, None
        if positions is not None:
            # PT uses all raw samples, including failed reconstructions, for stability.
            pairs, elements = self._coordinates(row, positions, atom_numbers)
        if mol is None:
            row['status'], row['error'] = input_status, input_error or 'No molecule'
            return None, row, None, None
        try:
            mol = clean_molecule(mol)
            row['smiles'] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if len(Chem.GetMolFrags(mol)) != 1:
                row['status'], row['error'] = 'disconnected', 'Disconnected molecular graph'
                return None, row, None, None
        except Exception as error:
            row['error'] = str(error)
            return None, row, None, None
        row['status'] = 'valid'
        self.molecules.append(mol)
        if positions is None and mol.GetNumConformers():
            pairs, elements = self._coordinates(row, mol.GetConformer().GetPositions(),
                                                [a.GetAtomicNum() for a in mol.GetAtoms()])
        if pairs is not None:
            self.pair_distances.extend(pairs)
            self.atom_types.update(int(z) for z in elements)
        try:
            if not mol.GetNumConformers() or not np.isfinite(mol.GetConformer().GetPositions()).all():
                raise ValueError('Finite input coordinates required for bond lengths')
            self.bond_distances.extend(eval_bond_length.bond_distance_from_mol(mol))
            row['bond_status'] = 'ok'
        except Exception as error:
            row['bond_status'], row['bond_error'] = 'failed', str(error)
        chem, vina = None, None
        try:
            chem = scoring_func.get_chem(mol)
            for key in ('qed', 'sa', 'sa_raw', 'logp', 'lipinski'):
                value = float(chem[key])
                if not np.isfinite(value):
                    raise ValueError(f'Nonfinite {key}')
                row[key] = value
            row['chemistry_status'] = 'ok'
        except Exception as error:
            row['chemistry_status'], row['chemistry_error'] = 'failed', str(error)
        if self.reference is not None:
            try:
                row['reference_tanimoto'] = tanimoto_sim(mol, self.reference)
                row['reference_status'] = 'ok'
            except Exception as error:
                row['reference_status'], row['reference_error'] = 'failed', str(error)
        if self.pli_scorer is not None:
            try:
                score = float(self.pli_scorer(mol))
                if not np.isfinite(score):
                    raise ValueError('Nonfinite PLI similarity')
                row['pli_similarity'], row['pli_status'] = score, 'ok'
            except Exception as error:
                row['pli_status'], row['pli_error'] = 'failed', str(error)
        if self.docking_scorer is not None:
            try:
                vina = self.docking_scorer(mol, docking_filename if docking_filename is not None else source)
                for key, results in vina.items():
                    if key == '_errors':
                        continue
                    metric = {'score_only': 'vina_score', 'minimize': 'vina_min', 'dock': 'vina_dock',
                              'qvina': 'qvina'}[key]
                    value = float(results[0]['affinity'])
                    if not np.isfinite(value):
                        raise ValueError(f'Nonfinite {metric}')
                    row[metric] = value
                row['docking_status'] = 'ok'
                if vina.get('_errors'):
                    row['docking_status'] = 'partial' if len(vina) > 1 else 'failed'
                    row['docking_error'] = json.dumps(vina['_errors'], sort_keys=True)
            except Exception as error:
                # Preserve QED, SA, PLI and geometry when docking fails.
                row['docking_status'], row['docking_error'] = 'failed', str(error)
        return mol, row, chem, vina

    def summary(self, num_attempts=None):
        n_input, n_valid = len(self.rows), len(self.molecules)
        if num_attempts is not None and (num_attempts < n_input or num_attempts < 1):
            raise ValueError('num_attempts must be positive and at least the number of input records')
        smiles = [row['smiles'] for row in self.rows if row['status'] == 'valid']
        reconstruction = [row['reconstructed'] for row in self.rows if row['reconstructed'] is not None]
        profile = eval_bond_length.get_bond_length_profile(self.bond_distances)
        pair_profile = eval_bond_length.get_pair_length_profile(self.pair_distances)
        diversity, diversity_error = None, ''
        try:
            diversity = pairwise_diversity(self.molecules, **self.fp_options)
        except Exception as error:
            diversity_error = str(error)
        return {
            'model': self.model, 'pocket_id': self.pocket_id, 'rdkit_version': rdBase.rdkitVersion,
            'protocol': self.protocol,
            'n_input_records': n_input, 'n_attempts': num_attempts,
            'n_unavailable_attempts': num_attempts - n_input if num_attempts is not None else None,
            'n_valid': n_valid, 'n_unique': len(set(smiles)),
            'validity': n_valid / num_attempts if num_attempts else None,
            'validity_among_inputs': n_valid / n_input if n_input else None,
            'uniqueness': len(set(smiles)) / n_valid if n_valid else None,
            'reconstruction_success': sum(reconstruction) / len(reconstruction) if reconstruction else None,
            'n_reconstruction_inputs': len(reconstruction),
            'diversity': {'mean': diversity, 'n_molecules': n_valid,
                          'n_pairs': n_valid * (n_valid - 1) // 2, **self.fp_options, 'error': diversity_error},
            'reference_fingerprint': {'kind': 'rdk', 'n_bits': 2048},
            'metrics': {key: finite_summary(row[key] for row in self.rows) for key in SCALAR_METRICS},
            'status_counts': {key: dict(Counter(row[key] for row in self.rows)) for key in (
                'status', 'chemistry_status', 'coordinate_status', 'stability_status', 'bond_status',
                'reference_status', 'pli_status', 'docking_status')},
            'stability': {'atom_stability': self.n_stable_atoms / self.n_atoms if self.n_atoms else None,
                          'molecule_stability': self.n_stable_mols / self.n_stability_mols if self.n_stability_mols else None,
                          'n_molecules': self.n_stability_mols, 'n_stable_molecules': self.n_stable_mols,
                          'n_atoms': self.n_atoms, 'n_stable_atoms': self.n_stable_atoms,
                          'coordinates_source': self.coordinates_source, 'hs': False},
            'geometry': {'bond_length_jsd': eval_bond_length.eval_bond_length_profile(profile),
                         'pair_distance_jsd': eval_bond_length.eval_pair_length_profile(pair_profile),
                         'atom_type_jsd': eval_atom_type.eval_atom_type_distribution(self.atom_types),
                         'n_bonds': len(self.bond_distances), 'n_pairs': len(self.pair_distances),
                         'reference': 'packaged TargetDiff empirical distributions',
                         'jsd_definition': 'Jensen-Shannon distance (sqrt divergence), natural logarithms'},
        }

    def write(self, output_dir, num_attempts=None, prefix=''):
        summary = self.summary(num_attempts)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f'{prefix}summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        if self.rows:
            with (output_dir / f'{prefix}molecules.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        return summary
