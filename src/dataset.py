import torch
from torch.utils.data import Dataset
import pandas as pd

# Pytorch dataset for all tweets with respective label
class TweetDataset(Dataset):
    def __init__(self, path, tokenizer):
        df = pd.read_csv(path)
        self.inputs = [tokenizer(str(s)) for s in df['content'].tolist()]
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input = self.inputs[index]
        label = self.labels[index]
        return input, torch.tensor(label)