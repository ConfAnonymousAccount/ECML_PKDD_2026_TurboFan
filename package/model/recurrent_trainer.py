import torch

def masked_mse(pred, target, lengths):

    mask = torch.arange(target.size(1))[None, :] < lengths[:, None]
    mask = mask.to(target.device)

    loss = (pred - target) ** 2
    loss = loss * mask.unsqueeze(-1)

    return loss.sum() / mask.sum()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RecurrentTrainer:
    def __init__(self,
                 model,
                 lr=1e-3,
                 criterion=None,
                 optimizer=None,
                 device=None,
                 scheduler=None,
                 scaler=None,
                 masked_loss=None):
        
        self.model = model
        self.device = device if device else model.device
        self.scaler = scaler
        self.model.to(self.device)

        self.train_losses = []
        self.val_losses = []

        # If masked_loss is provided, use it instead of criterion
        self.masked_loss = masked_loss

        if criterion is None and masked_loss is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

    # --------------------
    # TRAIN
    # --------------------
    def train_model(self, train_loader, epochs=10, val_loader=None):
        
        best_val_loss = float("inf")
        patience = 10
        counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for X, y, lengths in train_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(X, lengths)

                if self.masked_loss:
                    loss = self.masked_loss(outputs, y, lengths)
                else:
                    loss = self.criterion(outputs, y)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            self.train_losses.append(train_loss)

            if val_loader:
                val_loss = self.evaluate(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    self.save("best_model.pth")
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping triggered")
                        break

                if isinstance(self.scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler:
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']

                self.val_losses.append(val_loss)

                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train: {train_loss:.10f} "
                      f"Val: {val_loss:.10f} "
                      f"LR: {current_lr:.6f}")
            else:
                if self.scheduler:
                    self.scheduler.step()

                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train: {train_loss:.10f}")

    # --------------------
    # EVALUATE
    # --------------------
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X, y, lengths in data_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.model(X, lengths)

                if self.masked_loss:
                    loss = self.masked_loss(outputs, y, lengths)
                else:
                    loss = self.criterion(outputs, y)

                total_loss += loss.item()

        self.model.train()
        return total_loss / len(data_loader)

    # --------------------
    # PREDICT
    # --------------------
    def predict_loader(self, data_loader):

        self.model.eval()
        total_loss = 0.0

        observations = []
        predictions = []
        lengths_all = []
        
        with torch.no_grad():
            for X, y, lengths in data_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.model(X, lengths)

                if self.masked_loss:
                    loss = self.masked_loss(outputs, y, lengths)
                else:
                    loss = self.criterion(outputs, y)

                total_loss += loss.item()

                observations.append(y.cpu().numpy())
                predictions.append(outputs.cpu().numpy())
                lengths_all.extend(lengths.cpu().numpy())
        print(f"Loss: {total_loss / len(data_loader):.10f}")

        observations = np.vstack(observations)
        predictions = np.vstack(predictions)

        if self.scaler is not None:
            observations = [self.scaler.inverse_transform(observation)[:length] for observation, length in zip(observations, lengths_all)]
            predictions = [self.scaler.inverse_transform(prediction)[:length] for prediction, length in zip(predictions, lengths_all)]

        return observations, predictions

    # --------------------
    # SAVE
    # --------------------
    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    # --------------------
    # LOAD
    # --------------------
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.to(self.device)