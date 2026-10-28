"""Molecular graph similarity, separate from protein–ligand PLI similarity."""
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def fingerprint(mol, kind='rdk', radius=2, n_bits=2048, use_chirality=False):
    """Return an explicit bit fingerprint; input must be a sanitized molecule."""
    if mol is None or not mol.GetNumHeavyAtoms():
        raise ValueError('A nonempty molecule is required for fingerprinting')
    if radius < 0 or n_bits < 1:
        raise ValueError('radius must be nonnegative and n_bits positive')
    if kind == 'rdk':
        return Chem.RDKFingerprint(mol, fpSize=n_bits)
    if kind == 'morgan':
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=use_chirality)
        return generator.GetFingerprint(mol)
    raise ValueError(f'Unknown fingerprint: {kind}')


def tanimoto_sim(mol, ref, kind='rdk', **kwargs):
    """Tanimoto graph similarity. The legacy two-argument API uses RDKFingerprint."""
    return float(DataStructs.TanimotoSimilarity(
        fingerprint(mol, kind=kind, **kwargs), fingerprint(ref, kind=kind, **kwargs)))


def pairwise_diversity(molecules, kind='morgan', radius=2, n_bits=2048,
                       use_chirality=False):
    """Mean (1 - Tanimoto) over unordered sample pairs; None for fewer than two."""
    fps = [fingerprint(mol, kind=kind, radius=radius, n_bits=n_bits,
                       use_chirality=use_chirality) for mol in molecules]
    n_pairs = len(fps) * (len(fps) - 1) // 2
    if not n_pairs:
        return None
    similarity_sum = sum(sum(DataStructs.BulkTanimotoSimilarity(fp, fps[:idx]))
                         for idx, fp in enumerate(fps) if idx)
    return float(1 - similarity_sum / n_pairs)
