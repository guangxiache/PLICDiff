# PLICDiff

*Equivariant 3D Diffusion Model for Molecular Generation under Protein-Ligand Interaction Conditioning with Classifier-Free Guidance*

PLICDiff is a structure-based molecular generation model for protein pockets. It extends equivariant 3D diffusion generation with explicit protein-ligand interaction (PLI) conditioning extracted by PLIP, and uses classifier-free guidance during sampling to control generation toward the desired interaction pattern.

The data processing pipeline follows TargetDiff, while the model adds PLI-aware conditioning for pocket-specific ligand generation.

## Environment

Create the environment with micromamba or conda:

```bash
micromamba env create -f PLICDiff.yml
micromamba activate PLICDiff
```

Install the PyTorch and PyG wheels that match your CUDA version. The commands below reproduce the CUDA 11.8 setup used by this project:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

PLICDiff also uses RDKit, OpenBabel, PLIP, AutoDock Vina/QVina-related tooling, LMDB, PyTorch Geometric, and TensorBoard. Most Python dependencies are listed in `PLICDiff.yml`.

## Data

The data preparation procedure is based on [TargetDiff](https://arxiv.org/abs/2303.03543). See the [TargetDiff data instructions](https://github.com/guanjq/targetdiff?tab=readme-ov-file#data) for downloading and organizing CrossDocked data.

PLICDiff uses the CrossDocked protein-ligand complexes after:

1. RMSD filtering.
2. Pocket extraction within a 10 Angstrom radius around the ligand.
3. Train/validation/test splitting.
4. PLIP-based protein-ligand interaction feature extraction.

The expected default paths are configured in [configs/training.yml](configs/training.yml):

```text
data/crossdocked_v1.3_rmsd1.0_pocket10
data/crossdocked_pocket10_pose_split.pt
```

### Preprocessing

Clean CrossDocked pairs with RMSD filtering:

```bash
python scripts/data_preparation/clean_crossdocked.py \
  --source ./data/CrossDocked2020 \
  --dest ./data/crossdocked_v1.3_rmsd1.0 \
  --rmsd_thr 1.0
```

Extract 10 Angstrom binding pockets:

```bash
python scripts/data_preparation/extract_pockets.py \
  --source ./data/crossdocked_v1.3_rmsd1.0 \
  --dest ./data/crossdocked_v1.3_rmsd1.0_pocket10 \
  --radius 10 \
  --num_workers 16
```

Create dataset splits:

```bash
python scripts/data_preparation/split_pl_dataset.py \
  --path ./data/crossdocked_v1.3_rmsd1.0_pocket10 \
  --dest ./data/crossdocked_pocket10_pose_split.pt
```

## Checkpoint

The PLICDiff checkpoint is available on [Google Drive](https://drive.google.com/drive/folders/1sGeo79PXj7sLRuEs2fdutBS6k_yt7gar).

After downloading a checkpoint, update the checkpoint path in [configs/sampling.yml](configs/sampling.yml):

```yaml
model:
  checkpoint: ./path/to/checkpoint.pt
```

## Training

Train PLICDiff with:

```bash
python scripts/train.py \
  --config configs/training.yml \
  --device cuda \
  --logdir ./logs_diffusion \
  --tag plicdiff
```

Training logs, TensorBoard files, copied model code, and checkpoints are saved under `logs_diffusion/`.

To resume training:

```bash
python scripts/train.py \
  --config configs/training.yml \
  --device cuda \
  --resume_train_diffusion \
  --load_model_path ./logs_diffusion/path_to_run/checkpoints/checkpoint.pt
```

Important training options are controlled in `configs/training.yml`, including diffusion timesteps, beta schedules, model width/depth, batch size, validation frequency, optimizer, and scheduler.

## Sampling

Sampling is configured by `configs/sampling.yml`. The script can generate molecules for a target pocket, optionally using a reference ligand to extract PLI conditioning.

Example with a reference ligand:

```bash
python scripts/sample_for_pocket.py configs/sampling.yml \
  --pdb_path ./path/to/protein_or_pocket.pdb \
  --raw_ligand_path ./path/to/reference_ligand.sdf \
  --reference_ligand \
  --num_samples 200 \
  --batch_size 10 \
  --device cuda:0 \
  --result_path ./outputs/sample_pocket
```

The script saves:

- `results_<guidance_scale>.pt`: generated atom coordinates, atom types, trajectories, and reconstructed molecules.
- `sdf/`: reconstructed molecules in SDF format.
- `sample.yml`: a copy of the sampling configuration.

By default, the current sampling script uses guidance scale `1.3`. You can modify the `guidance_scale` list in `scripts/sample_for_pocket.py` to sample with multiple guidance strengths.

## Evaluation

Evaluate generated molecules with:

```bash
python scripts/evaluate_diffusion.py ./outputs/sample_pocket \
  --docking_mode none \
  --atom_enc_mode add_aromatic \
  --eval_step -1 \
  --sdf_path ./outputs/sample_pocket/sdf
```

Available docking modes are:

```text
none, qvina, vina_score, vina_dock
```

The evaluation script reports molecular stability, atom stability, reconstruction success, complete molecule rate, atom type distribution, bond length distribution, QED, SA, ring statistics, and optional docking scores. Results are saved under:

```text
outputs/sample_pocket/eval_results/
```

To evaluate how well generated ligands preserve the reference PLI pattern:

```bash
python scripts/calculate_plic_similarity.py \
  --row_ligand_fn ./path/to/reference_ligand.sdf \
  --ligand_dir ./outputs/sample_pocket/sdf \
  --protein_fn ./path/to/protein_or_pocket.pdb \
  --radius 10
```

This computes PLIP interaction features for generated ligands, compares them with the reference ligand by cosine similarity, and saves ranked similarity outputs together with top-scoring generated SDF files.


