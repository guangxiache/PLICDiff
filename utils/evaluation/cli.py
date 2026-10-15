"""Shared CLI options; optional PLIP and docking backends load only on request."""
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.evaluation.metrics import MoleculeMetrics, read_reference


def add_evaluation_arguments(parser, allow_qvina=False):
    parser.add_argument('--model', default='', help='Model label written to CSV/JSON.')
    parser.add_argument('--pocket_id', default='', help='Pocket label; evaluate one pocket per run.')
    parser.add_argument('--reference_ligand', help='Single-molecule reference SDF for graph similarity and/or PLI.')
    parser.add_argument('--diversity_radius', type=int, default=2)
    parser.add_argument('--diversity_bits', type=int, default=2048)
    parser.add_argument('--diversity_chirality', action='store_true')
    parser.add_argument('--pli', action='store_true', help='Calculate fixed-pocket PLI cosine similarity.')
    parser.add_argument('--pli_protein', help='Protein PDB for PLI; defaults to --protein_root if a PDB.')
    parser.add_argument('--pli_radius', type=float, default=10.0)
    modes = ['none', 'vina_score', 'vina_dock'] + (['qvina'] if allow_qvina else [])
    parser.add_argument('--docking_mode', choices=modes, default='none')
    parser.add_argument('--protein_root', help='Receptor PDB/PDBQT for Vina; dataset root for QVina.')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--docking_cpu', type=int, default=1)


def validate_evaluation_arguments(parser, args):
    if args.diversity_radius < 0 or args.diversity_bits < 1:
        parser.error('Diversity radius must be nonnegative and bits positive')
    if args.docking_cpu < 1 or args.exhaustiveness < 1:
        parser.error('Docking CPU and exhaustiveness must be positive')
    if args.docking_mode != 'none' and not args.protein_root:
        parser.error('Docking requires --protein_root')
    if args.pli:
        if args.pli_protein is None and args.protein_root and Path(args.protein_root).suffix.lower() == '.pdb':
            args.pli_protein = args.protein_root
        if not args.reference_ligand or not args.pli_protein:
            parser.error('--pli requires --reference_ligand and a protein PDB via --pli_protein')


class DockingRunner:
    def __init__(self, args):
        self.mode, self.protein = args.docking_mode, args.protein_root
        self.cpu, self.exhaustiveness = args.docking_cpu, args.exhaustiveness

    def __call__(self, mol, ligand_filename):
        if self.mode == 'qvina':
            from utils.evaluation.docking_qvina import QVinaDockingTask

            with TemporaryDirectory(prefix='plicdiff_qvina_') as directory:
                task = QVinaDockingTask.from_generated_mol(
                    mol, ligand_filename, protein_root=self.protein, tmp_dir=directory)
                return {'qvina': task.run_sync()}
        from utils.evaluation.docking_vina import VinaDockingTask

        with TemporaryDirectory(prefix='plicdiff_vina_') as directory:
            task = VinaDockingTask(self.protein, mol, tmp_dir=directory)
            modes = ['score_only', 'minimize'] + (['dock'] if self.mode == 'vina_dock' else [])
            results, errors = {}, {}
            for mode in modes:
                try:
                    results[mode] = task.run(mode=mode, cpu=self.cpu, exhaustiveness=self.exhaustiveness)
                except Exception as error:
                    errors[mode] = str(error)
            if errors:
                results['_errors'] = errors
            return results


def make_evaluator(args, output_dir, coordinates_source='input SDF conformers'):
    reference = read_reference(args.reference_ligand)
    pli = None
    if args.pli:
        from utils.evaluation.pli import FixedPocketPLI

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pli = FixedPocketPLI(args.reference_ligand, args.pli_protein,
                             Path(output_dir) / 'reference_pocket.pdb', args.pli_radius)
    evaluator = MoleculeMetrics(
        model=args.model, pocket_id=args.pocket_id, reference=reference, pli_scorer=pli,
        docking_scorer=DockingRunner(args) if args.docking_mode != 'none' else None,
        diversity_radius=args.diversity_radius, diversity_bits=args.diversity_bits,
        diversity_chirality=args.diversity_chirality, coordinates_source=coordinates_source)
    evaluator.protocol = dict(reference_ligand=args.reference_ligand, pli_enabled=args.pli,
                              pli_protein=args.pli_protein, pli_radius=args.pli_radius,
                              docking_mode=args.docking_mode, protein_root=args.protein_root,
                              docking_cpu=args.docking_cpu, exhaustiveness=args.exhaustiveness)
    return evaluator
