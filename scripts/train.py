import argparse
import os
import shutil

import numpy as np
import torch, gc
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model_pli_cross_atn import ScorePosNet3D


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--resume_train_diffusion', action='store_true', help='Whether to resume train model')
    parser.add_argument('--load_model_path', type=str) 
    args = parser.parse_args()

    # Load config
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    custom_name = 'test'
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, custom_name=custom_name) 
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transformations
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Dataset and DataLoader
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform
    )
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model construction
    logger.info('Building model...')
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)


    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            batch_size = batch.num_graphs
            batch_protein=batch.protein_element_batch
            pli = batch.pocket_interaction_condition
            pli = [torch.from_numpy(arr) for arr in pli]
            pli = torch.cat(pli, dim=0).cuda()
            pli = pli.to(torch.float32)
            # Set cfg
            drop_condition_prob = 0.3
            mask = torch.rand(batch_size, 1) > drop_condition_prob
            graph_sizes = torch.bincount(batch_protein).cpu()
            mask = mask.repeat_interleave(graph_sizes, dim=0).to(pli.device)
            pli = pli * mask.float()
            pli[:, -1] = torch.where(mask.squeeze(-1), pli[:, -1], torch.tensor(1.0, device=pli.device))

            # Add noise to protein positions
            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            results = model.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                pli = pli,
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch
            )
            loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()


        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()


    def validate(it):
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
        sum_loss_bond, sum_loss_non_bond = 0, 0
        all_pred_v, all_true_v = [], []
        all_pred_bond_type, all_gt_bond_type = [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)

                batch_size = batch.num_graphs
                batch_protein=batch.protein_element_batch
                pli = batch.pocket_interaction_condition
                pli = [torch.from_numpy(arr) for arr in pli]
                pli = torch.cat(pli, dim=0).cuda()
                pli = pli.to(torch.float32)
                # Set cfg
                drop_condition_prob = 0.3
                mask = torch.rand(batch_size, 1) > drop_condition_prob
                graph_sizes = torch.bincount(batch_protein).cpu()
                mask = mask.repeat_interleave(graph_sizes, dim=0).to(pli.device)
                pli = pli * mask.float()
                pli[:, -1] = torch.where(mask.squeeze(-1), pli[:, -1], torch.tensor(1.0, device=pli.device))

                t_loss, t_loss_pos, t_loss_v = [], [], []
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        pli=pli,
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step
                    )
                    loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']

                    # Accumulate losses
                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        # Calculate average losses
        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        # Calculate AUROC
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        # Update learning rate
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        # Log validation results
        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc
            )
        )
        # Log losses to TensorBoard
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.flush()
        return avg_loss


    try:
        if args.resume_train_diffusion:
                checkpoint = torch.load(args.load_model_path)
                model.load_state_dict(checkpoint['model'])
                # optimizer_vae.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                current_iter = checkpoint['iteration']
        else:
            current_iter = 1

        best_loss, best_iter = None, None
        for it in range(current_iter, config.train.max_iters + 1):
            train(it)
            if it <= 0.6*config.train.max_iters:
                if it % config.train.val_freq == 0 or it==1 or it==200:
                    val_loss = validate(it)
                    if best_loss is None or val_loss < best_loss:
                        logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                        best_loss, best_iter = val_loss, it
                        ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iteration': it,
                        }, ckpt_path)
                    else:
                        logger.info(f'[Validate] Val loss is not improved. '
                                    f'Best val loss: {best_loss:.6f} at iter {best_iter}')
            elif it > 0.6*config.train.max_iters and it <= 0.9*config.train.max_iters:
                if it % 1000 == 0:
                    val_loss = validate(it)
                    if best_loss is None or val_loss < best_loss:
                        logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                        best_loss, best_iter = val_loss, it
                        ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iteration': it,
                        }, ckpt_path)
                    else:
                        logger.info(f'[Validate] Val loss is not improved. '
                                    f'Best val loss: {best_loss:.6f} at iter {best_iter}')
            else:
                if it % 200 == 0 or it == config.train.max_iters:
                    val_loss = validate(it)
                    if best_loss is None or val_loss < best_loss:
                        logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                        best_loss, best_iter = val_loss, it
                        ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iteration': it,
                        }, ckpt_path)
                    else:
                        logger.info(f'[Validate] Val loss is not improved. '
                                    f'Best val loss: {best_loss:.6f} at iter {best_iter}')

    except KeyboardInterrupt:
        logger.info('Terminating...')
