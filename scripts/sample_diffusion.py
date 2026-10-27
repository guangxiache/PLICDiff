import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model_pli_cross_atn import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
from utils.runtime import load_checkpoint, project_path, validate_device



def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, data_raw_ligand, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior', pli=None, guidance_scale_pos=0, guidance_scale_v=0,
                            use_reference_ligand_in_guidance=True):
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError('num_samples and batch_size must be positive.')
    if (pos_only or sample_num_atoms == 'ref') and len(data.ligand_element) == 0:
        if data_raw_ligand is None:
            raise ValueError('pos_only/ref atom counts require a reference ligand.')
        data = data.clone()
        for key, value in data_raw_ligand:
            if key.startswith('ligand_'):
                data[key] = value
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
        if data_raw_ligand is not None:
            batch_raw_ligand = Batch.from_data_list([data_raw_ligand.clone() for _ in range(n_data)], follow_batch=['ligand_element']).to(device)
            raw_ligand_batch = batch_raw_ligand.ligand_element_batch
            ligand_v_reference = batch_raw_ligand.ligand_atom_feature_full
            ligand_pos_reference = batch_raw_ligand.ligand_pos
        else:
            batch_raw_ligand = None
            ligand_v_reference = None
            ligand_pos_reference = None
            raw_ligand_batch = None

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

            with torch.no_grad():
                r = model.sample_diffusion(
                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    pli = pli,
                    batch_protein=batch_protein,

                    init_ligand_pos=init_ligand_pos,
                    init_ligand_v=init_ligand_v,
                    batch_ligand=batch_ligand,
                    batch_raw_ligand=raw_ligand_batch,
                    ligand_v_reference=ligand_v_reference,
                    ligand_pos_reference=ligand_pos_reference,
                    num_steps=num_steps,
                    pos_only=pos_only,
                    center_pos_mode=center_pos_mode,
                    guidance_scale_pos=guidance_scale_pos,
                    guidance_scale_v=guidance_scale_v,
                    use_reference_ligand_in_guidance=use_reference_ligand_in_guidance
                )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--split_path', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--result_path', type=str, default='./outputs')
    parser.add_argument('--reference_ligand', action='store_true')
    parser.add_argument('--pdb_path', type=str)
    parser.add_argument('--raw_ligand_path', type=str)
    parser.add_argument('--no_pli', action='store_true', help='Disable the stored PLI condition and sample unconditionally')

    args = parser.parse_args()
    args.device = validate_device(args.device)
    args.config = project_path(args.config)
    if args.data_id is None or args.data_id < 0:
        parser.error('--data_id must be a nonnegative test-subset index')
    if args.reference_ligand:
        parser.error('For reference-ligand sampling use scripts/sample_for_pocket.py; '
                     'this entry point uses a stored dataset profile or --no_pli.')

    logger = misc.get_logger('sampling')

    # Load config
    config = misc.load_config(args.config)
    if args.checkpoint:
        config.model.checkpoint = project_path(args.checkpoint)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint or config.model.checkpoint, args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    ckpt['config'].data.path = project_path(args.data_path or ckpt['config'].data.path)
    ckpt['config'].data.split = project_path(args.split_path or ckpt['config'].data.split)
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    data = test_set[args.data_id]
    stored = getattr(data, 'pocket_interaction_condition', None)
    if stored is not None and not args.no_pli:
        pli = torch.as_tensor(np.asarray(stored), dtype=torch.float32, device=args.device)
        print(f'use stored pli profile from dataset, shape={tuple(pli.shape)}')
    else:
        pli = None

    # The dataset entry point uses stored PLI labels without reference-ligand geometry.
    all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, data_raw_ligand=None, num_samples=config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        pli = pli,
        guidance_scale_pos=getattr(config.sample, 'guidance_scale_pos', 3.0),
        guidance_scale_v=getattr(config.sample, 'guidance_scale_v', 3.0),
        use_reference_ligand_in_guidance=False,
    )
    result = {
        'data': data,
        'pred_ligand_pos': all_pred_pos,
        'pred_ligand_v': all_pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'sampling_metadata': {
            'guidance_convention': 'null + g * (conditional - null)',
            'guidance_scale_pos': config.sample.get('guidance_scale_pos', 3.0),
            'guidance_scale_v': config.sample.get('guidance_scale_v', 3.0),
            'use_reference_ligand_in_guidance': False,
            'data_id': args.data_id,
            'no_pli': args.no_pli,
        },
        # 'time': time_list
    }
    logger.info(f'Sample done! Guidance scale is pos={getattr(config.sample, "guidance_scale_pos", 3.0)}, v={getattr(config.sample, "guidance_scale_v", 3.0)}')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))
