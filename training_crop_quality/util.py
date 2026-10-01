import os
import glob
import torch
from torch.utils.data import Dataset
import random

class DocMDetectorDataset(Dataset):
    """Dataset loading precomputed bounding box quality samples for G_theta training."""

    def __init__(
        self,
        dataset_name_or_path=None,
    ):
        super().__init__()
        pattern = os.path.join(dataset_name_or_path, "*.pt")
        self.files = sorted(glob.glob(pattern))
        self.dataset_length=len(self.files)
        print(f"got {self.dataset_length} files")

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        retries=0
        while True:
            try :
                pt_path = self.files[idx]
                data = torch.load(pt_path)
                
                return data
            except :
                try :
                    os.remove(self.files[idx])
                except :
                    pass
                idx = random.randint(0, len(self.files) - 1)
                retries+=1
                print(f"[RETRY] Trying with a new random index for the {retries} time", flush=True) 
                
            
