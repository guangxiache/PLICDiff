# PLICDiff

PLICDiff is a protein–ligand interaction (PLI) conditioned diffusion framework for
3D ligand generation in protein binding pockets. It encodes PLIP-derived,
protein-atom-level interaction labels and uses classifier-free guidance (CFG)
to control their influence during sampling.

**Workflow:** reference complex → pocket and PLI profile → guided diffusion →
generated ligands → molecular and interaction evaluation.

## Contents

- [Installation](#installation)
- [Data and checkpoints](#data-and-checkpoints)
- [Generate molecules](#generate-molecules)
- [Guidance convention](#guidance-convention)
- [Evaluate molecules](#evaluate-molecules)
- [Training](#training)
- [Paths and repository layout](#paths-and-repository-layout)
- [Acknowledgments and license](#acknowledgments-and-license)

## Installation

Tested environment: Linux x86_64, Python 3.11, PyTorch 2.4.0, PyTorch Geometric 2.7.0,
RDKit 2023.09.1, Open Babel 3.1 and PLIP 2.3.0.

```bash
git clone https://github.com/guangxiache/PLICDiff.git
cd PLICDiff
micromamba env create -f PLICDiff.yml
micromamba activate plicdiff
```

Conda users can use `conda env create -f PLICDiff.yml` and `conda activate plicdiff`.

Install **one** PyTorch/PyG combination, then the remaining dependencies.

**CPU**

```bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-index torch-scatter==2.1.2 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
python -m pip install -r requirements.txt
```

**NVIDIA GPU — CUDA 11.8 wheels**

```bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-index torch-scatter==2.1.2 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
python -m pip install -r requirements.txt
```

PyTorch and its PyG extension wheels must match. See the official
[PyTorch commands](https://pytorch.org/get-started/previous-versions/#v240) and
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html).
[Docking dependencies](#optional-docking) are optional.

## Data and checkpoints

The preprocessed PLI-conditioned dataset and pretrained checkpoints are shared in
[Google Drive](https://drive.google.com/drive/folders/1sGeo79PXj7sLRuEs2fdutBS6k_yt7gar).

| Resource | Download / setup |
| --- | --- |
| Preprocessed dataset, with PLI included | Download `crossdocked_v1.3_rmsd1.0_pocket10_processed_final.lmdb` from the shared Google Drive folder. The uncompressed file is about **9.34 GiB**. |
| Matching train/validation/test split | Download `crossdocked_pocket10_pose_split.pt` from the same folder. |
| Pretrained checkpoints | Available in the same folder; default location: `checkpoint/9430000.pt`. |
| Generation inputs | Supply your own protein PDB and reference ligand SDF, or a prepared pocket PDB with its matching PLI profile. |

**The LMDB and its matching split are the two files needed to load the released
dataset for training and dataset-based sampling.** The LMDB already contains the
protein/ligand coordinates, graph and chemical features, and precomputed PLI
conditions in `pocket_interaction_condition`. Its seven channels are
`pipi, anion, cation, hbd, hba, hydro, none`, aligned to the protein atoms.
The loader reads these conditions directly, without rerunning PLIP for the cached
complexes. Raw pocket/ligand files, `index.pkl` and split-generation caches are
only needed for the corresponding preprocessing operations.

### Download and place the preprocessed data

From the repository root, create the data directory:

```bash
mkdir -p ./data/crossdocked_v1.3_rmsd1.0_pocket10
```

Place the downloaded files as follows:

```text
PLICDiff/
├── data/
│   ├── crossdocked_v1.3_rmsd1.0_pocket10/  # may remain empty when using the LMDB
│   ├── crossdocked_v1.3_rmsd1.0_pocket10_processed_final.lmdb
│   └── crossdocked_pocket10_pose_split.pt
└── checkpoint/
    └── 9430000.pt                        # for generation with pretrained weights
```

If you downloaded the compressed `.lmdb.gz` version, decompress it before use.
The compressed file is about **1.41 GiB**:

```bash
gzip -dk ./data/crossdocked_v1.3_rmsd1.0_pocket10_processed_final.lmdb.gz
```

Skip this command if you downloaded the uncompressed `.lmdb` file.
The default [training configuration](configs/training.yml) already uses:

```yaml
data:
  name: pl
  path: ./data/crossdocked_v1.3_rmsd1.0_pocket10
  split: ./data/crossdocked_pocket10_pose_split.pt
```

`data.path` names the pocket directory. The loader finds its sibling
`crossdocked_v1.3_rmsd1.0_pocket10_processed_final.lmdb` automatically.
Keep the supplied split paired with this LMDB and preserve the filenames.

SHA-256 checksums of the **uncompressed LMDB** and split:

```text
af9f236f683a633519c69fe04e76a1055de9fbe4d36dd8726f2b2e99c7529cf7  crossdocked_v1.3_rmsd1.0_pocket10_processed_final.lmdb
977ec123b51d8d9a03e2e007f50d1737103eaac06bc88fa4b81ee38d9dfae3d6  crossdocked_pocket10_pose_split.pt
```

For pretrained generation, set `model.checkpoint` in
[configs/sampling.yml](configs/sampling.yml), or pass `--checkpoint`.
The generation examples assume your own files are `inputs/protein.pdb` and
`inputs/reference.sdf`.

To rebuild the dataset yourself, follow the [optional raw-data preprocessing
instructions](#rebuild-from-raw-crossdocked-data-optional).

## Generate molecules

### Extract the PLI condition

The reference ligand must have 3D coordinates in the same coordinate frame as
the protein. Pockets use standard amino-acid `ATOM` records without hydrogens.

```bash
python scripts/extract_pli_profile.py \
  --protein ./inputs/protein.pdb --ligand ./inputs/reference.sdf \
  --cutoff 10 --out_dir ./conditions/example
```

This writes `pocket.pdb` and `pli_profile.npy`. The profile has shape
`(number_of_pocket_atoms, 7)`, with channels
`pipi, anion, cation, hbd, hba, hydro, none`. Donor/acceptor refers to the
**protein atom**. Keep the profile paired with its exported pocket and atom order.

### Sample ligands

```bash
python scripts/sample_for_pocket.py configs/sampling.yml \
  --checkpoint ./checkpoint/9430000.pt \
  --pdb_path ./conditions/example/pocket.pdb \
  --pli_path ./conditions/example/pli_profile.npy \
  --num_samples 50 --batch_size 5 --device cuda:0 \
  --guidance_scale 2.3 --result_path ./outputs/example
```

Use `--device cpu` for CPU execution. This example uses the current null-origin
coefficient `g=2.3`; see [Guidance convention](#guidance-convention).

| Output | Contents |
| --- | --- |
| `sample.yml` | Sampling configuration |
| `results_2.3.pt` | Final predictions, trajectories, reconstructed molecules and sampling metadata |
| `sampling_2.3.json` | Effective coefficients, reference-structure setting, seed and sampling parameters |
| `sdf/<sample_index>_2.3.sdf` | Successfully reconstructed, connected molecules |

The SDF count reflects successful, connected reconstructions. Re-running a guidance
setting replaces its SDF outputs. Use the full configured diffusion chain for
generation; `--num_steps` runs only its final steps for debugging.

For coordinate-only sampling, set `sample.pos_only: true` and provide a reference
ligand. This mode preserves the reference atom types and atom count.

### Other sampling inputs

Extract PLI directly from a reference complex during sampling:

```bash
python scripts/sample_for_pocket.py configs/sampling.yml \
  --pdb_path ./inputs/protein.pdb \
  --raw_ligand_path ./inputs/reference.sdf --reference_ligand \
  --num_samples 50 --batch_size 5 --device cuda:0 \
  --guidance_scale 2.3 --result_path ./outputs/example_reference
```

The reference extracts the PLI condition; whether its structure also enters the
encoder is controlled separately by `guidance_use_reference_ligand`.

For generation without PLI, provide a **prepared pocket** and omit both
`--pli_path` and `--reference_ligand`. A whole protein is not automatically cropped
in this mode. Multiple values, such as `--guidance_scale 1.0 1.3 2.3`, run a scale
sweep; evaluate each setting separately.

For a prepared dataset's test example:

```bash
python scripts/sample_diffusion.py configs/sampling.yml \
  --data_id 0 --device cuda:0 \
  --data_path ./data/crossdocked_v1.3_rmsd1.0_pocket10 \
  --split_path ./data/crossdocked_pocket10_pose_split.pt \
  --result_path ./outputs/test_0
```

This uses the stored PLI without reference-ligand structure. Add `--no_pli` to
remove the PLI condition.

## Guidance convention

The sampler predicts clean coordinates and atom-type logits. For either output:

```text
f_guided = f_null + g * (f_conditional - f_null)
```

`g=0` selects the null-PLI branch, `g=1` selects the conditional branch, and `g>1`
extrapolates. The alternative
`f_conditional + omega * (f_conditional - f_null)` is equivalent when **`g=1+omega`**:
for example, `omega=1.3` corresponds to `g=2.3`. Match the convention and conditioning
settings when comparing experimental GS labels.

The checked-in configuration uses `g=3.0`. `--guidance_scale` overrides both
coordinate/type coefficients; `--guidance_scale_pos` and `--guidance_scale_v` set
them separately. Guided logits pass through log-softmax before categorical sampling.

`guidance_use_reference_ligand: false` is the default: the encoder receives the
protein and PLI without reference-ligand structure. Setting it to `true` supplies
that structure to both CFG branches and requires
`--reference_ligand --raw_ligand_path`. Record this setting with the guidance scales.

## Evaluate molecules

See [**EVALUATION.md**](EVALUATION.md) for definitions, denominators, fingerprint
parameters, status fields and output schemas.

| Task | Entrypoint |
| --- | --- |
| PLICDiff PT predictions, reconstruction and raw-coordinate stability | [evaluate_diffusion.py](scripts/evaluate_diffusion.py) |
| SDF molecules from PLICDiff or another model | [evaluate_molecules.py](scripts/evaluate_molecules.py) |
| PLI-only scoring with sampler-style filename GS grouping | [calculate_plic_similarity.py](scripts/calculate_plic_similarity.py) |
| Paired comparison of pocket means | [compare_pocket_metrics.py](scripts/compare_pocket_metrics.py) |

### PLICDiff PyTorch (`.pt`) outputs

```bash
python scripts/evaluate_diffusion.py ./outputs/example \
  --model PLICDiff --pocket_id example \
  --atom_enc_mode add_aromatic --docking_mode none
```

Use the atom encoding from the checkpoint's training configuration. Results under
`outputs/example/eval_results/` include per-molecule CSV, summary JSON, legacy
metrics PT files and pair-distance histograms.

### SDF molecules, including PLI

Use the same entrypoint for every model, with **one model, pocket and guidance
setting per run**:

```bash
python scripts/evaluate_molecules.py \
  --sdf_dir ./outputs/example/sdf \
  --model PLICDiff --pocket_id example --num_attempts 50 \
  --reference_ligand ./inputs/reference.sdf \
  --pli --pli_protein ./inputs/protein.pdb --pli_radius 10 \
  --out_dir ./evaluation/PLICDiff/example
```

This writes `molecules.csv` and `summary.json`. Multi-record SDF files, explicit
SDF paths and a CSV `--manifest` are also supported.

| Metric group | Reported values |
| --- | --- |
| Molecular quality | Validity, uniqueness, diversity, QED, raw/transformed SA and additional descriptors |
| Structure similarity | Reference RDK fingerprint Tanimoto; Morgan-based pairwise diversity |
| Stability and geometry | Atom/molecule stability, bond-length, pair-distance and atom-type Jensen–Shannon distances |
| Interaction recovery | Fixed-pocket, six-channel PLI cosine similarity |
| Optional docking | Raw Vina Score, Vina Min and, if requested, redocking scores |

`--num_attempts` means **all generation attempts**, including failures and missing
outputs. If unknown, omit it: overall validity is `null`; validity among submitted
records is reported separately. Each scalar reports its successful count.

PLI uses one reference-defined pocket with fixed protein atom order and excludes
the `none` channel. Successful zero-contact analyses score 0; failed analyses have
no score and are reported separately. A reference with no interactions is rejected.
Optional docking failures do not discard chemistry, PLI or geometry results.

Transformed SA is `round((10 - SA_raw) / 9, 2)` per molecule, **higher is better**.
Raw Vina affinities are in kcal/mol, **lower (more negative) is better**.
Structure Tanimoto and diversity are distinct from PLI cosine similarity;
[similarity.py](utils/evaluation/similarity.py) provides the molecular fingerprint tools.

For PLI-only GS grouping, use `calculate_plic_similarity.py` with
`--reference_ligand`, `--protein_fn` and `--ligand_dir`. For baseline names without
the sampler's GS suffix, prefer `evaluate_molecules.py`.

### Paired pocket comparisons

After evaluating each model on the same set of pockets:

```bash
python scripts/compare_pocket_metrics.py \
  ./evaluation/PLICDiff/*/molecules.csv ./evaluation/Baseline/*/molecules.csv \
  --reference_model PLICDiff --metric pli_similarity \
  --n_bootstrap 20000 --seed 2024 --out_dir ./evaluation/comparison
```

Supply explicit `--model` and `--pocket_id` labels during evaluation. The comparison
computes **PLICDiff pocket mean minus baseline pocket mean**, with equal pocket
weights and a 95% paired pocket bootstrap interval. At least two pockets are needed;
every model must have successful scores in the same complete set of pockets.
Outputs are `pocket_means.csv`, `paired_differences.csv` and `comparison.json`.

### Optional docking

```bash
python -m pip install -r requirements-docking.txt
```

Add `--docking_mode vina_score --protein_root ./inputs/receptor.pdbqt --docking_cpu 1`
to either evaluation entrypoint. `vina_score` performs score-only evaluation and
local minimization; `vina_dock` additionally performs redocking.

A protein PDB input also requires `pdb2pqr` and a separately installed,
Python-compatible `prepare_receptor4.py`. Set `PLICDIFF_PREPARE_RECEPTOR` to its path
if it is not on PATH; `PLICDIFF_RECEPTOR_PYTHON` optionally selects its interpreter.
PLI requires a protein PDB even when Vina uses a prepared PDBQT.

The legacy PT evaluator also supports QVina, requiring `qvina2`/`qvina2.1` on PATH
or `PLICDIFF_QVINA`, plus `obabel` and receptor preparation. External tools are
installed separately. Meeko 0.5.1 and the existing 0.1 development API are supported.
Keep preparation and scoring settings consistent across models.

## Training

### Use the released preprocessed dataset

Follow [Download and place the preprocessed data](#download-and-place-the-preprocessed-data),
then run:

```bash
python scripts/train.py --config configs/training.yml \
  --device cuda --logdir ./logs_diffusion
```

The supplied LMDB and split support this workflow directly; you can skip the
raw-data preprocessing steps below. To use different locations, update `data.path`
and `data.split` in [configs/training.yml](configs/training.yml).
Validation uses `val`; `test` is reserved for evaluation. The split must have
nonempty `train` and `val` subsets. Use `--device cpu` for CPU execution.

`sample_time_method: symmetric` is the default timestep sampler. The `importance`
option uses a running squared-loss estimate and inverse-probability loss weights.

### Resume training

Resume the model, optimizer and scheduler together:

```bash
python scripts/train.py --config configs/training.yml --device cuda \
  --resume_train_diffusion \
  --load_model_path ./logs_diffusion/RUN/checkpoints/ITERATION.pt
```

### Rebuild from raw CrossDocked data (optional)

Obtain CrossDocked structures following the
[TargetDiff data instructions](https://github.com/guanjq/targetdiff#data), then run:

```bash
python scripts/data_preparation/clean_crossdocked.py \
  --source ./data/CrossDocked2020 \
  --dest ./data/crossdocked_v1.3_rmsd1.0 --rmsd_thr 1.0

python scripts/data_preparation/extract_pockets.py \
  --source ./data/crossdocked_v1.3_rmsd1.0 \
  --dest ./data/crossdocked_v1.3_rmsd1.0_pocket10 --radius 10 --num_workers 4

python scripts/data_preparation/split_pl_dataset.py \
  --path ./data/crossdocked_v1.3_rmsd1.0_pocket10 \
  --dest ./data/crossdocked_pocket10_pose_split.pt --num_cpu 4
```

When the processed LMDB is absent, the dataset's first initialization builds it
and computes the PLI conditions. The split script initializes the dataset, so
this computation can occur while running that script. Use a separate data/output
directory when building a new dataset or split. A TargetDiff LMDB without PLICDiff
interaction features cannot be substituted for the supplied PLI-conditioned LMDB.

## Paths and repository layout

Examples run from the repository root. Entry scripts can also be invoked by
absolute path from another directory after activating the environment.

| Path type | Resolution |
| --- | --- |
| YAML `data.path`, `data.split` and `model.checkpoint` | Relative to the repository root |
| CLI configuration paths, `--checkpoint`, `--data_path` and `--split_path` | Relative to the repository root |
| CLI protein, ligand, profile and output paths | Relative to the current working directory |
| Relative SDF paths in a manifest | Relative to the manifest's directory |

Configuration paths support absolute paths, `~` and environment variables.

```text
configs/                    Training and sampling configurations
datasets/                   Protein–ligand data and shared PLIP extraction
models/                     Diffusion model and geometric networks
scripts/PlicEncoder.py      Graph-attention PLI encoder
scripts/data_preparation/   CrossDocked preparation
scripts/                    Training, sampling, extraction and evaluation CLIs
utils/evaluation/           Chemistry, similarity, stability, geometry and docking
utils/                      Featurization, reconstruction and runtime helpers
EVALUATION.md               Detailed metric definitions and examples
PLICDiff.yml                Chemistry environment
requirements.txt            Core Python dependencies
requirements-docking.txt    Optional Vina dependencies
```

Download data and pretrained weights separately using the links above.

## Acknowledgments and license

PLICDiff builds on [TargetDiff](https://github.com/guanjq/targetdiff) and uses
[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric),
[RDKit](https://www.rdkit.org/), [Open Babel](https://openbabel.org/) and
[PLIP](https://github.com/pharmai/plip).

License texts: [PLICDiff](LICENSE), [TargetDiff](third_party/TargetDiff_LICENSE),
and [LiGAN](third_party/LiGAN_LICENSE). The reconstruction module contains
LiGAN-derived code under GPLv2; its source notice is retained in
[reconstruct.py](utils/reconstruct.py). Other third-party notices remain in the
corresponding source files.
