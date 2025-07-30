import os
import re
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class DevBenchDataset(Dataset):
    """
    Dataset from a given dataset folder.
    -----
    Constructor inputs: 
    - dataset_folder: the path to the folder containing the dataset
      and manifest
    - manifest_file: the location of the manifest CSV within the 
      dataset folder; should contain one row per trial, with *relative*
      paths to any images
    """
    def __init__(self, dataset_folder, manifest_file="manifest.csv"):
        self.dataset_folder = dataset_folder
        self.manifest = pd.read_csv(os.path.join(dataset_folder, manifest_file))
        self.num_image_cols = len([c for c in self.manifest.columns if re.compile("image[0-9]").match(c)])
        self.num_text_cols = len([c for c in self.manifest.columns if re.compile("text[0-9]").match(c)])
    
    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        images = []
        for i in range(1, self.num_image_cols + 1):
            image_path = os.path.join(self.dataset_folder, row[f"image{i}"])
            with Image.open(image_path).convert('RGB') as img:
                images.append(img.copy())  # Copy the image data to memory
        texts = [row[f"text{i}"] for i in range(1, self.num_text_cols + 1)]
        return {"images": images, "text": texts}

def collator(batch):
    """
    Collates data, turning it from a list of dicts to a dict of lists.
    -----
    Inputs: 
    - batch: a list of dicts with the same keys for each dict
    Outputs:
    - a dict of lists
    """
    return {key: [item for ex in batch for item in ex[key]] for key in batch[0]}

def make_dataloader(dataset):
    """
    Constructs a dataloader from a dataset.
    -----
    Inputs: 
    - dataset: an object of class Dataset, with additional attribute
      trial_type
    Outputs:
    - an object of class Dataloader
    """
    batch_size = 1 if dataset.num_image_cols > 1 or dataset.num_text_cols > 1 else 16
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return dl