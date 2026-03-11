import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, ShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def get_package_root():
    """
    Returns the absolute path to the root of the package.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SequenceDataset(Dataset):
    """
    sequences: list of tensors/arrays, each shape (T_i, F)
    metadata: list of dicts with aligned tensors of shape (T_i, ...)
    """
    def __init__(self, sensors, indicators, metadata):
        assert len(sensors) == len(metadata)
        assert len(indicators) == len(metadata)
        assert len(indicators) == len(sensors)

        self.data = []
        item = {}
        for sensors_, indicators_, meta in zip(sensors, indicators, metadata):
            # seq = torch.tensor(seq, dtype=torch.float32)
            sensors_ = torch.tensor(sensors_, dtype=torch.float64)
            indicators_ = torch.tensor(indicators_, dtype=torch.float64)
            T = sensors_.shape[0]

            item["sensors"] = sensors_
            item["indicators"] = indicators_

            proc_meta = {}
            for k, v in meta.items():
                v = torch.tensor(v) if not torch.is_tensor(v) else v
                if v.shape[0] != T:
                    raise ValueError(f"Metadata '{k}' length mismatch.")
                proc_meta[k] = v

            item.update(proc_meta)
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def pad_collate_fn(batch):
    """
    Pads sequences and all aligned metadata on time axis.
    """

    output = {}
    keys = batch[0].keys()

    for key in keys:
        tensors = [item[key] for item in batch]

        padded = pad_sequence(tensors, batch_first=True)
        output[key] = padded

    return output


def create_kfold_splits(
        n_sequences: int,
        k_folds: int=5,
        train_ratio: float=0.7,
        val_ratio: float=0.15,
        test_ratio: float=0.15,
        random_state: int=42):
    """
    Generator yielding (train_set, val_set, test_set) for each fold.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    indices = np.arange(n_sequences)

    ss = ShuffleSplit(n_splits=k_folds, 
                      test_size=1-train_ratio, 
                      random_state=random_state)

    for train_idx, valtest_idx in ss.split(indices):

        # Split val/test inside the fold
        total_vt = len(valtest_idx)
        val_size = int(total_vt * val_ratio / (val_ratio + test_ratio))

        val_idx = valtest_idx[:val_size]
        test_idx = valtest_idx[val_size:]
        
        yield train_idx, val_idx, test_idx

def create_sequence_datasets(sequences, metadata, train_idx, val_idx, test_idx):
    # Create dataset objects for the fold
    train_set = SequenceDataset(
        [np.asarray(sequences[i][:, 1:8], dtype=float) for i in train_idx],
        [np.asarray(sequences[i][:, -13:-3], dtype=float) for i in train_idx],
        [metadata[i] for i in train_idx]
    )
    val_set = SequenceDataset(
        [np.asarray(sequences[i][:, 1:8], dtype=float)  for i in val_idx],
        [np.asarray(sequences[i][:, -13:-3], dtype=float) for i in val_idx],
        [metadata[i] for i in val_idx]
    )
    test_set = SequenceDataset(
        [np.asarray(sequences[i][:, 1:8], dtype=float) for i in test_idx],
        [np.asarray(sequences[i][:, -13:-3], dtype=float) for i in test_idx],
        [metadata[i] for i in test_idx]
    )

    return train_set, val_set, test_set

def create_dataloaders_from_datasets(
        train_set,
        val_set,
        test_set,
        batch_size=1,
        pad_batches=True,
        shuffle_train=True,
        num_workers=0):

    collate_fn = pad_collate_fn if pad_batches else None

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=shuffle_train, num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
        
if __name__ == "__main__":
    import os
    import pandas as pd
    
    file_name = "sequences_with_info.csv"
    data_path = os.path.join(get_package_root(), "data", file_name)
    data = pd.read_csv(data_path)
    input_features = list(data.columns[2:9])
    output_features = list(data.columns[-13:-3])
    print(data.shape)
    
    sequences = (
        data.groupby("sequence_id", sort=False)[data.columns[1:]]
            .apply(np.asarray)   # each group (DataFrame) -> ndarray
            .tolist()
    )

    metadata = [
        {
            "maintenance": np.asarray(sequence[:, -3], dtype=bool),
        }
        for sequence in sequences
    ]

    for train_idx, val_idx, test_idx in create_kfold_splits(n_sequences=len(sequences),
                                                            k_folds=5,
                                                            train_ratio=0.7,
                                                            val_ratio=0.1,
                                                            test_ratio=0.2):
                
        train_set, val_set, test_set = create_sequence_datasets(sequences, metadata, train_idx, val_idx, test_idx)
        train_loader, val_loader, test_loader = create_dataloaders_from_datasets(train_set, val_set, test_set,
                                                                                 batch_size=4,
                                                                                 pad_batches=False
                                                                                )

        batch = next(iter(train_loader))

        print(batch["sensors"].shape)
        print(batch["indicators"].shape)
        print(batch["maintenance"].shape)
        break
    