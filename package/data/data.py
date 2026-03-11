import os
import pickle
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset

from OpenDeckGeneration.data_cv import create_flatten_datasets

OUTPUT_FEATURES = ['deg_CmpBst_s_mapEff_in', 'deg_CmpBst_s_mapWc_in', 'deg_CmpFan_s_mapEff_in', 
                   'deg_CmpFan_s_mapWc_in', 'deg_CmpH_s_mapEff_in', 'deg_CmpH_s_mapWc_in', 
                   'deg_TrbH_s_mapEff_in', 'deg_TrbH_s_mapWc_in', 'deg_TrbL_s_mapEff_in', 
                   'deg_TrbL_s_mapWc_in']

def collate_fn(batch):

    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    lengths = torch.tensor([len(seq) for seq in inputs])

    padded_inputs = pad_sequence(inputs, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)

    return padded_inputs, padded_targets, lengths

class SequenceDataset(Dataset):
    def __init__(self, sequences, indices,
                 input_indices,
                 output_indices,
                 input_scaler=None,
                 output_scaler=None):

        self.sequences = sequences
        self.indices = indices
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        seq = self.sequences[self.indices[idx]]  # (length, dim)

        X = seq[:, self.input_indices]
        y = seq[:, self.output_indices]

        # Normalize if scalers provided
        if self.input_scaler is not None:
            X = torch.tensor(
                self.input_scaler.transform(X),
                dtype=torch.float32
            )

        if self.output_scaler is not None:
            y = torch.tensor(
                self.output_scaler.transform(y),
                dtype=torch.float32
            )

        return X, y

def fit_scalers(sequences, train_idx, input_indices, output_indices):

    X_all = []
    y_all = []

    for i in train_idx:
        seq = sequences[i]
        X_all.append(seq[:, input_indices])
        y_all.append(seq[:, output_indices])

    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)

    input_scaler = StandardScaler().fit(X_all)
    output_scaler = StandardScaler().fit(y_all)

    return input_scaler, output_scaler

class Data:
    def __init__(self):
        super().__init__()
        self.df = None
        self.sequences = None
        self.size = 0
        
        self.trajectories = None
        self.measures_series = None
        self.trajectory_len = None
        
        self.scaler = None
        self.loaded = False
    
    def load(self, load_path):
        self.df = pd.read_csv(load_path)
        values = (
            self.df
            .groupby("sequence_id")[list(self.df.columns[1:])]
            .apply(lambda x: x.to_numpy())
            .to_list()
         )

        self.sequences = np.array(values, dtype="O")
            
        self.size = len(self.sequences)  
        self.loaded = True      
    
    def create_loader_flatten(self, train_idx, val_idx, test_idx, batch_size=128, shuffle_train=True, scale_inputs=True, scale_outputs=False):
        if not self.loaded:
            raise Exception("The data should be imported first")
        train_dataset, val_dataset, test_dataset, scaler = create_flatten_datasets(self.sequences, 
                                                                                   train_idx, 
                                                                                   val_idx, 
                                                                                   test_idx, 
                                                                                   shuffle_train=shuffle_train,
                                                                                   scale_inputs=scale_inputs,
                                                                                   scale_outputs=scale_outputs)
        train_sensors, train_indicators = train_dataset
        val_sensors, val_indicators = val_dataset
        test_sensors, test_indicators = test_dataset
        
        train_dataset = TensorDataset(torch.tensor(train_sensors), torch.tensor(train_indicators))
        val_dataset = TensorDataset(torch.tensor(val_sensors), torch.tensor(val_indicators))
        test_dataset = TensorDataset(torch.tensor(test_sensors), torch.tensor(test_indicators))
                
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader, scaler
    
    def create_dataloaders_series(self,
                                  train_idx,
                                  val_idx,
                                  test_idx,
                                  input_indices,
                                  output_indices,
                                  batch_size=32):

        # Fit scalers on training data
        input_scaler, output_scaler = fit_scalers(
            self.sequences,
            train_idx,
            input_indices,
            output_indices
        )

        # Create datasets
        train_dataset = SequenceDataset(
            self.sequences, train_idx,
            input_indices, output_indices,
            input_scaler, output_scaler
        )

        val_dataset = SequenceDataset(
            self.sequences, val_idx,
            input_indices, output_indices,
            input_scaler, output_scaler
        )

        test_dataset = SequenceDataset(
            self.sequences, test_idx,
            input_indices, output_indices,
            input_scaler, output_scaler
        )

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        return train_loader, val_loader, test_loader, input_scaler, output_scaler
