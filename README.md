# PLICDiff

A PLI-conditioned 3D diffusion framework for protein-pocket ligand generation with classifier-free guidance (CFG). PLIP-derived protein-atom interaction labels condition a graph-attention encoder and a geometric diffusion backbone.

## Installation

The code has been checked on Linux x86_64 with Python 3.11, PyTorch 2.4.0, PyG 2.7.0, RDKit 2023.09.1 and Open Babel 3.1. Windows and macOS installations have not been validated; use Linux or WSL2 for the documented workflow. Data and pretrained checkpoints are downloaded separately.

```bash
git clone https://github.com/guangxiache/PLICDiff.git
cd PLICDiff
micromamba env create -f PLICDiff.yml
micromamba activate plicdiff
```

Conda can be used instead of micromamba. Then select **one** PyTorch installation matching the machine.

CPU:

```bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-index torch-scatter==2.1.2 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
python -m pip install -r requirements.txt
```

NVIDIA GPU, CUDA 11.8 wheels:

```bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-index torch-scatter==2.1.2 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
python -m pip install -r requirements.txt
```

The installed PyTorch and PyG extension wheels must match. See the official [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v240) and [PyG wheel instructions](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html). `torchvision`, `torchaudio`, Jupyter and a separate property-prediction model are not needed by the PLICDiff workflow.

## Paths and repository layout

```text
configs/          Training and sampling configurations
models/           Diffusion model and geometric networks
scripts/          Training, sampling, PLI extraction and evaluation
scripts/data_preparation/  CrossDocked preparation
scripts/PlicEncoder.py    Graph-attention PLI encoder (needed by the model)
datasets/         Protein-ligand graph data and shared PLIP extraction
utils/            Featurization, reconstruction and metric utilities
PLICDiff.yml      Chemistry environment
requirements.txt  Core Python dependencies
requirements-docking.txt  Optional docking dependencies
```

Paths in `configs/*.yml`, including `data.path`, `data.split` and `model.checkpoint`, are relative to the repository root, independent of the current working directory. Absolute paths, `~` and environment variables are also supported. Configuration-file arguments, `--checkpoint`, and the dataset overrides `--data_path`/`--split_path` are also resolved relative to the repository root. Use absolute paths for files stored elsewhere. Protein, ligand and profile inputs and output directories supplied on the command line are relative to the current working directory unless absolute. No personal installation path is required.

Run examples from the repository root, or invoke the scripts by their absolute paths from elsewhere. Activate the environment before running them.

## Data preparation and training

The preprocessing workflow follows [TargetDiff](https://github.com/guanjq/targetdiff); obtain CrossDocked data using its [data instructions](https://github.com/guanjq/targetdiff#data).

```bash
python scripts/data_preparation/clean_crossdocked.py \
  --source ./data/CrossDocked2020 --dest ./data/crossdocked_v1.3_rmsd1.0 --rmsd_thr 1.0
python scripts/data_preparation/extract_pockets.py \
  --source ./data/crossdocked_v1.3_rmsd1.0 \
  --dest ./data/crossdocked_v1.3_rmsd1.0_pocket10 --radius 10 --num_workers 4
python scripts/data_preparation/split_pl_dataset.py \
  --path ./data/crossdocked_v1.3_rmsd1.0_pocket10 \
  --dest ./data/crossdocked_pocket10_pose_split.pt --num_cpu 4
```

The first access to an unprocessed dataset builds its LMDB and PLIP conditions; full-dataset preparation is a substantial offline step. The protein pocket must contain standard amino-acid `ATOM` records without hydrogens. Reference ligands must have 3D coordinates in the same coordinate frame as the protein. PLI rows follow the protein atom order. A TargetDiff LMDB without PLICDiff interaction features cannot be used directly; build the conditions from raw pocket/ligand pairs with this project.

Update `configs/training.yml` and run:

```bash
python scripts/train.py --config configs/training.yml --device cuda --logdir ./logs_diffusion
```

Use `--device cpu` for CPU execution. The split must contain nonempty `train` and `val` subsets. Validation uses `val`; `test` is reserved for evaluation. The earlier training script used `test` for validation, so a new training run has a corrected validation protocol. Resume model, optimizer and scheduler together with:

```bash
python scripts/train.py --config configs/training.yml --device cuda \
  --resume_train_diffusion --load_model_path ./logs_diffusion/RUN/checkpoints/ITERATION.pt
```

## Checkpoints and guidance convention

The existing checkpoint download is linked from [Google Drive](https://drive.google.com/drive/folders/1sGeo79PXj7sLRuEs2fdutBS6k_yt7gar). Set `model.checkpoint` in `configs/sampling.yml`, or pass `--checkpoint` to a sampler. The default location is `checkpoint/9430000.pt`. Checkpoints contain Python configuration objects and should come from a trusted source.

Current sampling combines clean-coordinate predictions and atom-type logits as:

```text
f_guided = f_null + g * (f_conditional - f_null)
```

`g=0` selects the null-PLI branch, `g=1` selects the conditional branch, and `g>1` extrapolates. Conditional-origin coefficients `omega` in `f_conditional + omega*(f_conditional - f_null)` correspond to `g=1+omega`. Historical GS labels must be converted using the convention of the code that produced them. The checked-in configuration uses `g=3.0`; `--guidance_scale` overrides it. Separate coordinate/type coefficients are supported through configuration or `--guidance_scale_pos` and `--guidance_scale_v`. The effective values are saved with each sampling result.

`guidance_use_reference_ligand: false` gives the conditioning encoder only the protein and PLI labels. Setting it to `true` also supplies the reference ligand structure to both CFG branches. These are different conditioning settings; record the setting when comparing experiments. Encoder branch probability remains the existing default of 0.6 for the complex path during training; this is separate from the 0.3 PLI dropout.

## Sampling

Extract a reusable PLI profile and its matching pocket:

```bash
python scripts/extract_pli_profile.py --ligand reference.sdf --protein protein.pdb --out_dir ./conditions/example
python scripts/sample_for_pocket.py configs/sampling.yml \
  --pdb_path ./conditions/example/pocket.pdb --pli_path ./conditions/example/pli_profile.npy \
  --num_samples 50 --batch_size 5 --device cuda:0 --guidance_scale 2.3 \
  --result_path ./outputs/example
```

The profile shape is `(number_of_pocket_atoms, 7)` with channel order `pipi, anion, cation, hbd, hba, hydro, none`. In `hbd`/`hba`, donor/acceptor refers to the protein atom. Profiles and their pocket files must stay paired. Extraction uses the requested `--cutoff` (default 10 Å).

Alternatively, extract PLI directly from a reference ligand during sampling:

```bash
python scripts/sample_for_pocket.py configs/sampling.yml \
  --pdb_path protein.pdb --raw_ligand_path reference.sdf --reference_ligand \
  --num_samples 50 --batch_size 5 --device cuda:0 --guidance_scale 2.3 \
  --result_path ./outputs/example_reference
```

To generate without PLI, supply only a pocket through `--pdb_path` and omit `--reference_ligand` and `--pli_path`. A whole protein is not automatically cropped in this mode. A scale sweep accepts multiple values, e.g. `--guidance_scale 1.0 1.3 2.3`. Each result includes trajectories, reconstructed molecules, a `sampling_*.json` record and SDF files for reconstructed connected molecules. For profile extraction, the pocket is written to the output directory; protein/ligand source files are not modified.

Use the full configured diffusion chain for generation. `--num_steps` can truncate the chain for debugging; it does not implement an accelerated sampler and a shortened run is not a fully denoised sample.

Dataset sampling is also available:

```bash
python scripts/sample_diffusion.py configs/sampling.yml --data_id 0 --device cuda:0 \
  --data_path ./data/crossdocked_v1.3_rmsd1.0_pocket10 \
  --split_path ./data/crossdocked_pocket10_pose_split.pt --result_path ./outputs/test_0
```

It uses the test example's stored PLI profile without reference-ligand structure. Add `--no_pli` for the null-PLI setting. The path overrides are useful for checkpoints created on another machine.

## Evaluation

Chemistry, reconstruction, stability and distance-distribution metrics do not require a docking installation:

```bash
python scripts/evaluate_diffusion.py ./outputs/example --docking_mode none --atom_enc_mode add_aromatic
```

PLI similarity is evaluated on a **single reference-defined pocket**, keeping the protein atom order fixed across all generated ligands:

```bash
python scripts/calculate_plic_similarity.py --reference_ligand reference.sdf \
  --protein_fn protein.pdb --ligand_dir ./outputs/example/sdf --radius 10
```

The six interaction channels are flattened for cosine similarity; the `none` channel is excluded. A successfully analyzed ligand with no interactions scores zero. Failed analyses are excluded from the mean and reported separately in CSV/JSON, together with success/failure counts. A reference with no detected interactions is rejected as uninformative. The old `--row_ligand_fn` argument remains an alias. Filenames without a GS suffix are included in group `all`.

### Optional docking

```bash
python -m pip install -r requirements-docking.txt
```

Vina accepts a prepared receptor PDBQT without AutoDockTools:

```bash
python scripts/evaluate_diffusion.py ./outputs/example --docking_mode vina_score \
  --protein_root receptor.pdbqt
```

With a protein PDB input, the legacy receptor-preparation route requires `pdb2pqr` and a separately installed Python-compatible `prepare_receptor4.py`. Set `PLICDIFF_PREPARE_RECEPTOR` to that script if it is not on PATH; `PLICDIFF_RECEPTOR_PYTHON` optionally selects its interpreter. QVina additionally requires `qvina2`/`qvina2.1` on PATH or `PLICDIFF_QVINA`, and Open Babel's `obabel`. No hardcoded `adt` Conda environment is activated. Install external tools separately; their source trees are not included in this repository. Meeko 0.5.1 uses its supported RDKit API; the existing 0.1 development API is retained for compatibility, and docking values can vary with preparation-tool versions.

## Acknowledgments

PLICDiff builds on [TargetDiff](https://github.com/guanjq/targetdiff) and uses [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), [RDKit](https://www.rdkit.org/), [Open Babel](https://openbabel.org/) and [PLIP](https://github.com/pharmai/plip). Retain the included upstream copyright notices when redistributing the code. The repository does not include training data, pretrained weights, paper drafts, generated molecules or local analysis outputs.
