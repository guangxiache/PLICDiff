#!/usr/bin/env python3
"""从 (配体, 蛋白) 对提取相互作用 profile，存为 .npy —— de novo 生成的条件接口。

训练时模型按概率见过 (pocket-only, 真实 pli) 组合（PlicEncoder prob_if=0.6、
train.py drop_condition_prob=0.3），因此采样可以不给参考配体、只给相互作用
profile（7 维 one-hot，每行对应一个口袋原子）。本脚本把任意已知配体与蛋白的
相互作用提取成该 profile，供 scripts/sample_for_pocket.py --pli_path 使用；
也可手工编辑 .npy 实现"可编程相互作用"。

用法：
    python scripts/extract_pli_profile.py \
        --ligand <参考配体.sdf> --protein <蛋白.pdb> \
        --out_dir <输出目录>

产物：
    <out_dir>/pli_profile.npy   # (口袋原子数, 7) float32
    <out_dir>/pocket.pdb        # 与 profile 逐原子对应的口袋（采样时传给 --pdb_path）

7 维顺序（与训练一致）：pipi, anion, cation, hbd, hba, hydro, none
"""
import argparse
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.extract_PLIC import PLIC  # noqa: E402

INTERACTION_TYPES = ['pipi', 'anion', 'cation', 'hbd', 'hba', 'hydro', 'none']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ligand', type=str, required=True, help='参考配体 SDF（只用于提取 profile，采样时不喂给模型）')
    parser.add_argument('--protein', type=str, required=True, help='完整蛋白 PDB（内部按 cutoff 重切口袋）')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--cutoff', type=float, default=10.0)
    args = parser.parse_args()

    for fn in (args.ligand, args.protein):
        if not os.path.exists(fn):
            sys.exit(f'文件不存在: {fn}')

    os.makedirs(args.out_dir, exist_ok=True)
    pocket_out = os.path.join(args.out_dir, 'pocket.pdb')
    plic = PLIC()
    # already_cropped=False：从完整蛋白重切口袋；return_pocket=True：拿到与 profile
    # 逐原子对应的口袋文件（与 sample_for_pocket --reference_ligand 路径完全一致）
    pli, pocket_path = plic.pli_processor(
        args.ligand, args.protein, already_cropped=False, return_pocket=True,
        cutoff=args.cutoff, pocket_output=pocket_out)
    if pli is None:
        sys.exit('profile 提取失败（pli_processor 返回 None），请检查配体/蛋白文件')

    os.makedirs(args.out_dir, exist_ok=True)
    npy_path = os.path.join(args.out_dir, 'pli_profile.npy')
    np.save(npy_path, np.asarray(pli, dtype=np.float32))

    pli = np.asarray(pli)
    counts = {t: int(pli[:, i].sum()) for i, t in enumerate(INTERACTION_TYPES)}
    print(f'profile: {npy_path}  shape={pli.shape}')
    print(f'pocket:  {pocket_out}')
    print('相互作用分布（每口袋原子一行 one-hot）: ' + ', '.join(f'{k}={v}' for k, v in counts.items()))
    print('采样示例：')
    print(f'  python scripts/sample_for_pocket.py configs/sampling.yml '
          f'--pdb_path {pocket_out} --pli_path {npy_path} --result_path <out>')


if __name__ == '__main__':
    main()
