import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from multiprocessing import Pool

from utils.data import PDBProtein, parse_sdf_file
from .pl_data import ProteinLigandData, torchify_dict
from . import extract_PLIC as extract_PLIC


class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, num_cpu=8, transform=None, version='final'):
        super().__init__()  
        self.raw_path = raw_path.rstrip('/')  
        self.index_path = os.path.join(self.raw_path, 'index.pkl')  
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb') 
        self.transform = transform  
        self.db = None  
        self.keys = None  
        self.num_cpu = num_cpu

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _connect_db(self):   
        assert self.db is None, 'A connection has already been opened.'  # Ensure no existing connection
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,  # Do not create new database
            subdir=False,
            readonly=True,  # Read-only mode
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()  
        self.db = None
        self.keys = None

    
    def process_item(self, item): 
        i, (pocket_fn, ligand_fn, *_) = item  # Unpack
        if pocket_fn is None:
            return 
        try:
            data_prefix = self.raw_path 

            pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
            ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))

            data = ProteinLigandData.from_protein_ligand_dicts( 
                protein_dict=torchify_dict(pocket_dict), 
                ligand_dict=torchify_dict(ligand_dict),
            )
            # Calculate protein-ligand interaction conditions
            PLI = extract_PLIC.PLIC()
            protein_raw_path = os.path.join(data_prefix, pocket_fn)
            ligand_raw_path = os.path.join(data_prefix, ligand_fn)
            pocket_pli_cond = PLI.pli_processor(ligand_raw_path, protein_raw_path)
            assert pocket_pli_cond is not None, "Protein-ligand interaction condition calculation failed, skipping data processing." 

            data.protein_filename = pocket_fn
            data.ligand_filename = ligand_fn
            data.pocket_interaction_condition = pocket_pli_cond
            assert len(data.pocket_interaction_condition)==len(data.protein_pos), "Protein interaction atom count does not match protein coordinate atom count, skipping data processing."
            data = data.to_dict() 
            return (i, data)  
        
        except Exception as e:
            print(f'error: {e}')
            return None

    def _process(self):

        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True, 
            subdir=False,
            readonly=False,  
        )


        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)


        with db.begin(write=True, buffers=True) as txn: 
            with Pool(processes = self.num_cpu) as pool:
                for result in tqdm(pool.imap_unordered(self.process_item, enumerate(index)), total=len(index)):
                    if result is not None:
                        i, data = result
                        # if i == 0:
                        #     print(data[pocket_interaction_condition].shape)
                        txn.put(
                            key=str(i).encode(),  
                            value=pickle.dumps(data)  
                        )
        db.close() 


    
    def __len__(self):
        if self.db is None:
            self._connect_db() 
        return len(self.keys) 


    def __getitem__(self, idx):
        data = self.get_ori_data(idx)  
        if self.transform is not None:
            data = self.transform(data)  
        return data
    
    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db() 
        key = self.keys[idx] 
        data = pickle.loads(self.db.begin().get(key))  
        data = ProteinLigandData(**data) 
        data.id = idx 
        
        try:
            if data.protein_pos.size(0) <= 0:  
                raise ValueError("Protein position information does not exist, skipping data processing.")
        except ValueError as ve:
            print(f"Error occurred while processing index {idx}: {ve}")
            return None 
        
        return data

    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str) 
    parser.add_argument('--num_cpu',  type=int, default=8)
    args = parser.parse_args()


    dataset = PocketLigandPairDataset(args.path, args.num_cpu)
    print(len(dataset), dataset[0])
