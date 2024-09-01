import torch
from torch.utils.data import DataLoader


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: list[dict[str, list]]):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        return item

    def __len__(self):
        return len(self.encodings)


def create_dataloader(dat: list[dict[str, list]], batch_size: int) -> DataLoader:
    return DataLoader(TestDataset(dat), batch_size=batch_size, shuffle=False)
