import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from package.data import Data
from package.model import MLP
from OpenDeckGeneration.data_cv import create_kfold_splits

def objective(trial, train_loader, val_loader, device="cuda:0"):

    # ------------------------
    # Sample Hyperparameters
    # ------------------------
    n_layers = trial.suggest_int("n_layers", 1, 4)

    hidden_sizes = []
    for i in range(n_layers):
        hidden_sizes.append(
            trial.suggest_int(f"n_units_l{i}", 32, 256, step=32)
        )

    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # ------------------------
    # Build Model
    # ------------------------

    model = MLP(
        input_size=28,
        hidden_sizes=tuple(hidden_sizes),
        output_size=10,
        dropout=dropout,
        device=device
    )
    model.to(float)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.MSELoss()  # or MSELoss for regression

    # ------------------------
    # Training Loop (Short)
    # ------------------------

    epochs = 20
    best_val_loss = float("inf")

    for epoch in range(epochs):

        # ---- Train ----
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Report to Optuna (for pruning)
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss

if __name__ == "__main__":
    file_name = "2000_aligned_and_clean_indexfalse.csv"
    file_path = os.path.join("data", file_name)
    data = Data()
    data.load(load_path=file_path)
    
    for train_idx, val_idx, test_idx in create_kfold_splits(n_sequences=len(data.sequences),
                                                        k_folds=5,
                                                        train_ratio=0.7,
                                                        val_ratio=0.1,
                                                        test_ratio=0.2):
        train_loader, val_loader, test_loader, scaler = data.create_loader_flatten(train_idx,
                                                                                val_idx,
                                                                                test_idx,
                                                                                batch_size=128,
                                                                                shuffle_train=True,
                                                                                scale_inputs=True,
                                                                                scale_outputs=True)
        break

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner()
                                )

    study.optimize(lambda trial: objective(trial, train_loader, val_loader), 
                   n_trials=50)

    print("Best trial:")
    print(study.best_trial.params)
    