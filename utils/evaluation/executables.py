"""Locate optional docking tools without a machine-specific Conda environment."""
import importlib.util
import os
from pathlib import Path
import shutil
import sys


def find_executable(name, environment_variable=None, alternatives=()):
    override = os.environ.get(environment_variable) if environment_variable else None
    for candidate in ([override] if override else []) + [name, *alternatives]:
        path = shutil.which(candidate)
        if path:
            return path
        path = Path(sys.prefix) / 'bin' / candidate
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise FileNotFoundError(f'{name} is not installed or not executable. '
                            f'Add it to PATH' + (f' or set {environment_variable}.' if environment_variable else '.'))


def receptor_preparation_command():
    path = os.environ.get('PLICDIFF_PREPARE_RECEPTOR') or shutil.which('prepare_receptor4.py')
    if not path:
        spec = importlib.util.find_spec('AutoDockTools')
        if spec and spec.submodule_search_locations:
            path = str(Path(next(iter(spec.submodule_search_locations))) / 'Utilities24' / 'prepare_receptor4.py')
    if not path or not Path(path).is_file():
        raise FileNotFoundError('Receptor preparation requires prepare_receptor4.py. '
                                'Set PLICDIFF_PREPARE_RECEPTOR, or supply a prepared receptor PDBQT to Vina.')
    return [os.environ.get('PLICDIFF_RECEPTOR_PYTHON', sys.executable), path]
