"""PLI cosine similarity on one fixed, reference-defined protein pocket."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rdkit import Chem


def cosine_similarity(x, y):
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f'PLI atom alignment mismatch: {x.shape} versus {y.shape}')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite PLI features')
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(np.dot(x, y) / norm, -1.0, 1.0)) if norm else 0.0


class FixedPocketPLI:
    def __init__(self, reference_ligand, protein, pocket_output, radius=10.0):
        from datasets.extract_PLIC import PLIC

        if radius <= 0 or not np.isfinite(radius):
            raise ValueError('PLI pocket radius must be finite and positive')
        self.plic = PLIC()
        profile, self.pocket_path = self.plic.pli_processor(
            str(reference_ligand), str(protein), already_cropped=False,
            return_pocket=True, cutoff=radius, pocket_output=Path(pocket_output))
        if profile is None:
            raise RuntimeError('Reference PLI extraction failed; no scores were assigned')
        self.reference = profile[:, :6]
        if not np.any(self.reference):
            raise ValueError('Reference has no interactions; PLI cosine is not informative')

    def score_file(self, ligand_path):
        profile = self.plic.pli_processor(str(ligand_path), self.pocket_path, already_cropped=True)
        if profile is None:
            raise ValueError('PLIP/ligand parsing failed')
        return cosine_similarity(self.reference, profile[:, :6])

    def __call__(self, mol):
        if not mol.GetNumConformers():
            raise ValueError('PLI scoring requires input 3D coordinates')
        with TemporaryDirectory(prefix='plicdiff_pli_ligand_') as directory:
            path = Path(directory) / 'ligand.sdf'
            with Chem.SDWriter(str(path)) as writer:
                writer.write(mol)
            return self.score_file(path)
