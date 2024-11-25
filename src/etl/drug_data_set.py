import torch
from prepare_dataset import y_train, y_val
from tokenization import train_encodings, val_encodings

class DrugDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = DrugDataset(train_encodings, y_train.values)
val_dataset = DrugDataset(val_encodings, y_val.values)