from __future__ import annotations

import polars as pl

from Pil import Image
from torch.utils.data import Dataset, DataLoader
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class DevBenchDataset(Dataset):
    """Dataset for devbench tasks. This can be image based
    tasks or a mix of images and text.
    """

    def __init__(self: DevBenchDataset, data_folder_path: Path, manifest_file: str = "manifest.csv") -> None:
        """The constructor of a DevBenchDataset.

        Args:
            data_folder_path(Path): The path to the directory
                containing all the data necessary for the
                creation of the dataset.
            manifest_file(str): The name of the file that
                defines an item of the dataset.
        """
        self.data_folder_path: Path = data_folder_path
        self.uid: str = data_folder_path.stem
        manifest: pl.DataFrame = pl.read_csv(data_folder_path / manifest_file)
        num_image_cols: int = len([c for c in self.manifest.columns if "image" in c])
        num_text_cols: int = len([c for c in self.manifest.columns if "text" in c])

        self.data: list[dict[str, list[Image] | list[str]]] = []
        for item in manifest.iter_rows(named=True):
            self.data.append(self._parse_item(item), num_image_cols, num_text_cols)

    def _parse_item(self: DevBenchDataset, item: dict[str, str], num_image_cols: int, num_text_cols: int) -> dict[str, list[Image] | list[str]]:
        images = []
        for i in range(1, num_image_cols + 1):
            image_path = self.data_folder_path / item[f"image{i}"]
            with Image.open(image_path).convert("RGB") as img:
                images.append(img.copy())
        texts = [item[f"text{i}"] for i in range(1, num_text_cols + 1)]
        return {"images": images, "text": texts}

    def __len__(self: DevBenchDataset) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self: DevBenchDataset, idx: int) -> dict[str, list[Image] | list[str]]:
        """Returns an element of the dataset.

        Args:
            idx(int): The index of the element to return.

        Returns:
            dict[str, list[Image] | list[str]]: A dictionary
                containing two lists, one of images and one of
                texts.
        """
        return self.data[idx]


def collate_fn(batch: tuple[dict[str, list[Image] | list[str]]]) -> dict[str, list[Image] | list[str]]:
    """A function that takes a tuple of dicts and collates them
    into a dict of lists.

    Args:
        batch(tuple): A tuple of size batch size with dicts as
            elements.

    Returns:
        dict[str, list[Image] | list[str]]: A dictionary of two
            lists, one containing the images, the other the
            texts.
    """
    return {key: [item for ex in batch for item in ex[key]] for key in batch[0]}


def make_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Constructs the DataLoader from a dataset to enable
    multi-process batching.

    Args:
        dataset(Dataset): The dataset from which to create the
            dataloader
        batch_size(int): The batch size of the DataLoader.

    Returns:
        DataLoader: The DataLoader of the dataset passed.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader
