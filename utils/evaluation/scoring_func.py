from collections import Counter
from copy import deepcopy

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.QED import qed

from utils.evaluation.sascorer import calculateScore, compute_sa_score as compute_sa_score

__all__ = ['obey_lipinski', 'get_logp', 'get_chem', 'get_molecule_force_field',
           'calculateScore', 'compute_sa_score']


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def get_logp(mol):
    return Crippen.MolLogP(mol)


def get_chem(mol):
    qed_score = qed(mol)
    # Match compute_sa_score: score a canonicalized graph, then round per molecule.
    sa_raw = calculateScore(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
    sa_score = round((10 - sa_raw) / 9, 2)
    logp_score = get_logp(mol)
    lipinski_score = obey_lipinski(mol)
    ring_info = mol.GetRingInfo()
    ring_size = Counter([len(r) for r in ring_info.AtomRings()])

    return {
        'qed': qed_score,
        'sa': sa_score,
        'sa_raw': sa_raw,
        'logp': logp_score,
        'lipinski': lipinski_score,
        'ring_size': ring_size
    }


def get_molecule_force_field(mol, conf_id=None, force_field='mmff', **kwargs):
    """Build UFF or MMFF94/MMFF94s for the selected conformer."""
    conf_id = -1 if conf_id is None else conf_id
    force_field = force_field.lower()
    if force_field == 'uff':
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=conf_id, **kwargs)
    elif force_field in {'mmff', 'mmff94', 'mmff94s'}:
        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(
            mol, mmffVariant='MMFF94s' if force_field == 'mmff94s' else 'MMFF94')
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=conf_id, **kwargs)
    else:
        raise ValueError("Invalid force_field {}".format(force_field))
    return ff
