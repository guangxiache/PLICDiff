from math import pi
import os
import argparse
import random
import torch
from tqdm.auto import tqdm
# import multiprocessing
import pickle

from torch.utils.data import Subset
from datasets.pl_pair_dataset_mp import PocketLigandPairDataset


def get_chain_name(fn):
    return os.path.basename(fn)[:6]  

def get_pdb_name(fn):
    return os.path.basename(fn)[:4]


def get_unique_pockets(dataset, raw_id, used_pdb, num_pockets):
    # only save first encountered id for unseen pdbs
    unique_id = []
    pdb_visited = set()
    for idx in tqdm(raw_id, 'Filter'):
        pdb_name = get_pdb_name(dataset[idx].ligand_filename)  
        if pdb_name not in used_pdb and pdb_name not in pdb_visited:
            unique_id.append(idx)
            pdb_visited.add(pdb_name)

    print('Number of Pairs: %d' % len(unique_id))
    print('Number of PDBs:  %d' % len(pdb_visited))

    random.Random(args.seed).shuffle(unique_id)
    unique_id = unique_id[:num_pockets]  
    print('Number of selected: %d' % len(unique_id))
    return unique_id, pdb_visited.union(used_pdb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data/crossdocked_v1.3_rmsd1.0_pocket10')
    parser.add_argument('--dest', type=str, default='./data/crossdocked_pocket10_pose_split.pt')
    parser.add_argument('--train', type=int, default=200000)
    parser.add_argument('--val', type=int, default=1000)
    parser.add_argument('--test', type=int, default=10000)
    parser.add_argument('--val_num_pockets', type=int, default=-1)
    parser.add_argument('--test_num_pockets', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_cpu', type=int, default=12)
    parser.add_argument('--allowed_ids_path', type=str, default='./data/allowed_ids.pt')
    parser.add_argument('--train_pdb_info_path', type=str, default='./data/train_pdb_info.pkl')
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path, args.num_cpu)
    print('Load dataset successfully!')

    if True:
        allowed_elements = {1, 6, 7, 8, 9, 15, 16, 17}
        elements = {i: set() for i in range(90)}  
        if not os.path.exists(args.allowed_ids_path):
            print(f'{args.allowed_ids_path} does not exist, begin processing data')
            valid_ids = set()  
            for i, data in enumerate(tqdm(dataset, desc='Filter')):
                if data is None:
                    continue
                elif data.ligand_filename is None:    
                    print(f'Warning: data with index {i} has no ligand filename!')
                    continue
                elif not len(data.pocket_interaction_condition)==len(data.protein_pos):
                    print(f'Warning: data with index {i} is wrong with atom number of PLI !')
                    continue
                valid_ids.add(i) 
                for e in data.ligand_element:
                    elements[e.item()].add(i) 
            # all_id = set(range(len(dataset)))
            blocked_id = set().union(*[ 
                elements[i] for i in elements.keys() if i not in allowed_elements  
            ])
            allowed_id = list(valid_ids - blocked_id)
            random.Random(args.seed).shuffle(allowed_id)   
            with open(args.allowed_ids_path, 'wb') as f:
                pickle.dump(allowed_id, f)            
        else:
            with open(args.allowed_ids_path, 'rb') as f:
                allowed_id = pickle.load(f)
        print('Allowed: %d' % len(allowed_id))

        if not os.path.exists(args.train_pdb_info_path):
            train_id = allowed_id[:args.train]  
            train_set = Subset(dataset, indices=train_id)  
            train_pdb = {get_pdb_name(d.ligand_filename) for d in tqdm(train_set)}   
            with open(args.train_pdb_info_path, 'wb') as f: 
                pickle.dump(train_pdb, f)
            print('train pdb info saved! File path: ', args.train_pdb_info_path, 'PDBs: ', len(train_pdb))
        else:
            with open(args.train_pdb_info_path, 'rb') as f: 
                train_pdb = pickle.load(f)
            print('train pdb info loaded! File path: ', args.train_pdb_info_path, 'PDBs: ', len(train_pdb))

        if args.val_num_pockets == -1:
            # not group by pocket
            val_id = allowed_id[args.train: args.train + args.val]
            used_pdb = train_pdb
        else:
            raw_val_id = allowed_id[args.train: args.train + args.val]
            val_id, used_pdb = get_unique_pockets(dataset, raw_val_id, train_pdb, args.val_num_pockets)   

        if args.test_num_pockets == -1:
            test_id = allowed_id[args.train + args.val: args.train + args.val + args.test]
        else:
            raw_test_id = allowed_id[args.train + args.val: args.train + args.val + args.test]
            test_id, used_pdb = get_unique_pockets(dataset, raw_test_id, used_pdb, args.test_num_pockets)

    torch.save({   
        'train': train_id,  # 200000
        'val': val_id,   # 1000
        'test': test_id,  # 100
    }, args.dest)

    print('Train %d, Validation %d, Test %d.' % (len(train_id), len(val_id), len(test_id)))
    print('Done.')
