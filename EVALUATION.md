# Evaluation metrics

Evaluate **one model and one pocket per run**. Keep each pocket's protein, reference ligand,
generated coordinates, model label and generation settings paired. Use the same protocol for
every model. The scripts calculate metrics; they do not regenerate the paper's experiments.

## Entrypoints

| Input / purpose | Entrypoint |
| --- | --- |
| PLICDiff sampling trajectories or final predictions (`results_*.pt`, `result_*.pt`) | `scripts/evaluate_diffusion.py` |
| SDF molecules from PLICDiff or a baseline | `scripts/evaluate_molecules.py` |
| Standalone PLI scoring with historical filename-based GS grouping | `scripts/calculate_plic_similarity.py` |
| Paired comparison of pocket means | `scripts/compare_pocket_metrics.py` |

All CLI input/output paths are relative to the current working directory unless absolute.
PT files contain Python objects; load only sampling outputs from trusted sources.
The SDF entrypoint does not require PyTorch, PLIP or docking to calculate chemistry metrics.

```bash
python scripts/evaluate_diffusion.py ./outputs/pocket_A \
  --model PLICDiff --pocket_id pocket_A --docking_mode none

python scripts/evaluate_molecules.py --sdf_dir ./baseline/pocket_A \
  --model Baseline --pocket_id pocket_A --num_attempts 50 \
  --out_dir ./evaluation/Baseline/pocket_A
```

`--sdf_dir` reads only `*.sdf` directly in that directory. It does not search subdirectories
or infer GS groups. Do not mix pockets or guidance settings in one run. Alternatively pass
explicit SDF paths as positional arguments. Every SDF record is evaluated, including
duplicates and invalid records in multi-molecule files.

To include expected but missing files, use `--manifest inputs.csv` instead. The CSV has
`path` and optional unique `sample_id` columns. Relative paths are resolved against the
manifest's directory. A missing/empty file is recorded as one failed input; invalid records
are retained in the counts. Duplicate input paths/IDs are rejected to prevent double counting.
`--num_attempts` still refers to generation attempts, not files, and must cover all input records.

## Outputs and denominators

The SDF entrypoint writes `molecules.csv` and `summary.json`. The PT entrypoint writes
`<eval_step>_<input_stem>_molecules.csv` / `..._summary.json` per sampling file,
`summaries_<eval_step>.json`, the legacy `metrics_<eval_step>_<input_stem>.pt`, and
pair-distance histograms. Its existing CLI options and PT keys `stability`, `bond_length`,
`all_results` remain available; each PT output also contains `summary`.

CSV rows include parsing/validity, coordinate, stability, chemistry, PLI and docking status
and errors. Summary JSON reports each scalar's successful count, mean and median.
Empty metrics are `null`, not zero. Optional metrics that were not requested are identified.
Docking failures do not remove valid molecules from QED, SA, diversity, PLI or geometry.

| Metric | Implementation / denominator |
| --- | --- |
| Validity | Sanitized, nonempty, connected heavy-atom graphs / **all generation attempts**. PT attempts are counted from stored predictions. SDF requires `--num_attempts`; otherwise overall validity is `null`. `validity_among_inputs` always refers only to submitted records. |
| Reconstruction success | PT predictions that return an RDKit molecule / PT predictions. Connectivity and chemical validity are checked separately. Not inferable from retained SDF files; SDF reports `null`. |
| Uniqueness | Number of distinct canonical **isomeric** SMILES / number of valid records. Explicit H is removed. Stereoisomers are distinct; duplicates remain in all other sample-based metrics. No tautomer/protonation standardization is performed. |
| Diversity | Mean `1 - Tanimoto` over all unordered pairs of valid records, including duplicate pairs. Default Morgan bit fingerprint: radius 2, 2048 bits, chirality off. Fewer than two valid records gives `null`. |
| QED | RDKit QED; mean over successfully scored valid molecules. |
| SA raw | Ertl–Schuffenhauer SA score from the included scorer and `fpscores.pkl.gz`; lower is easier. |
| SA (`sa`) | `round((10 - SA_raw) / 9, 2)` **per molecule before averaging**; higher is easier. Kept compatible with the original `compute_sa_score`. |
| Reference Tanimoto | Optional molecular **structure** similarity to `--reference_ligand`, using RDKit RDKFingerprint (2048 bits). It is separate from PLI cosine and from Morgan diversity. |
| Atom stability | Number of stable atoms / atoms in successfully evaluated coordinate arrays. |
| Molecule stability | Arrays with all atoms stable / successfully evaluated coordinate arrays. |
| Bond-length / pair-distance JSD | SciPy Jensen–Shannon **distance**, square root of divergence, natural logarithms, against the packaged TargetDiff empirical histograms. Missing bond types / empty pair histograms give `null`. |
| Atom-type JSD | Jensen–Shannon distance to the packaged atom-type reference. Empty observations or atoms outside its support give `null`. |
| PLI similarity | Optional six-channel fixed-pocket cosine, over successfully analyzed valid molecules, including successful zero-contact scores. |
| Vina Score / Vina Min | Optional raw Vina score-only / local minimization affinity, kcal/mol, **lower (more negative) is better**. Each reports its successful count. |

The paper does not specify all historical diversity fingerprint parameters. The explicit
defaults here make new calculations reproducible; they are not evidence that older diversity
values used identical parameters. Override with `--diversity_radius`, `--diversity_bits`,
`--diversity_chirality` and report the settings.

Stability follows `utils/evaluation/analyze.py`: distances infer bond orders using its
element-pair thresholds and margins; with `hs=False`, an atom is stable when
`0 < inferred bond-order sum <= allowed_bonds[element]`. Supported elements are
H, C, N, O, F, P, S and Cl. This heuristic is distinct from RDKit sanitization.
Unsupported elements, missing/nonfinite coordinates and their coverage are reported.
PT stability uses raw coordinates **before reconstruction**, including failed reconstructions.
SDF stability uses supplied conformers after H removal; it cannot recover pre-reconstruction
stability or failures omitted by another model. Compare the same representation across models.
Bond and pair histograms use valid connected molecules with available coordinates; each
distance is an observation, rather than each molecule or pocket receiving equal weight.
These are histogram comparisons, not proofs of conformer strain, stereochemistry or activity.

## Fixed-pocket PLI

```bash
python scripts/evaluate_molecules.py --sdf_dir ./baseline/pocket_A \
  --model Baseline --pocket_id pocket_A --num_attempts 50 \
  --reference_ligand reference.sdf --pli --pli_protein protein.pdb --pli_radius 10 \
  --out_dir ./evaluation/Baseline/pocket_A
```

The reference ligand and protein must share a coordinate frame with every generated ligand.
PLIP extracts one reference-defined pocket; its protein atom order stays fixed for every
ligand. The six scored channels are `pipi, anion, cation, hbd, hba, hydro`, excluding `none`.
The extractor retains one prioritized interaction label per atom, using that channel order;
donor/acceptor describes the protein atom. Flatten both aligned profiles and calculate cosine.
A successful zero-contact profile scores 0. A parsing/PLIP failure has no score and is
excluded from the mean, with failure counts/errors retained. A reference without detected
interactions is rejected. PLI is calculated on the submitted pose, before optional docking.
Both PLI entrypoints use `utils/evaluation/pli.py` and `datasets/extract_PLIC.py`.

## Optional Vina

Install `requirements-docking.txt` and the receptor-preparation tools described in README.
Add these options to either evaluation entrypoint:

```bash
--docking_mode vina_score --protein_root receptor.pdbqt --docking_cpu 1
```

`vina_score` runs score-only and local minimization separately. `vina_dock` also performs
redocking, with `--exhaustiveness 16` by default. The legacy PT `qvina` mode needs its
external tools and dataset-relative `data.ligand_filename`; it uses QVina's existing
UFF preparation and is a different protocol. Record receptor preparation, box settings,
software versions and pose provenance when comparing models. PDBQT is for Vina; PLI
requires a PDB via `--pli_protein`. Docking imports are lazy when `--docking_mode none`.

## Paired pocket comparison

Use CSV outputs carrying explicit `--model` and `--pocket_id` labels. For example, expand
only the result CSVs from the selected models and pockets:

```bash
python scripts/compare_pocket_metrics.py \
  ./evaluation/PLICDiff/*/molecules.csv ./evaluation/Baseline/*/molecules.csv \
  --reference_model PLICDiff --metric pli_similarity --n_bootstrap 20000 --seed 2024 \
  --out_dir ./evaluation/comparison
```

The script first averages successful scores **within each model/pocket**, then calculates
`PLICDiff pocket mean minus baseline pocket mean`. It averages those paired differences
with equal pocket weights and reports a 95% percentile bootstrap interval by resampling
paired pockets (20,000 draws), rather than treating all molecules as independent targets.
All models must have successful scores in the same complete set of pockets; missing pairs
and duplicate samples are rejected. Outputs are `pocket_means.csv`, `paired_differences.csv`
and `comparison.json`. Positive differences favor the reference for PLI; for raw Vina,
negative differences favor it. There is no automatic reversal or normalization of scores.
