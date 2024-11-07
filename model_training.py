import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from chess_model_v1 import *


class EarlyStopping:
    def __init__(self, patience=3, delta=0, save_path="best_model.pth"):

        # patience = epochs tolerated since last validation loss improved
        # delta = minimum change in loss to qualify as improvement

        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        # Check if the current validation loss is the best we've seen
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)  # Save the best model
            print(f"validation loss improved, model saved to {self.save_path}")
        else:
            self.counter += 1
            print(f"validation loss did not improve. Patience counter: {self.counter}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("early stopping triggered")


class ChessDataset(Dataset):
    def __init__(self, file_path, transform=None):
        data = np.load(file_path)
        self.features = data["features"]
        self.labels = data["labels"]

        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        board_state = self.features[idx]
        move_label = self.labels[idx]

        if self.transform:
            board_state = self.transform(board_state)

        board_state_tensor = torch.tensor(board_state, dtype=torch.float32)
        move_label_tensor = torch.tensor(move_label, dtype=torch.float32)



        return board_state_tensor, move_label_tensor
    

    

def train_model(model, train_loader, num_epochs, learning_rate=0.001,  batch_size=64):

    model.train()
    loss_fn = nn.MSELoss() # mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #early_stopping = EarlyStopping(patience=3, delta=0, save_path=save_path)

    for epoch in range(num_epochs):
        
        total_loss = 0.0
        model.train()

        for batch_index, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)

            loss = loss_fn(outputs.squeeze(), labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if batch_index % 100 == 99:
                print(f"Epoch: [{epoch+1}/{num_epochs}], Batch: [{batch_index+1}], Loss: [{total_loss / 100:.4f}]")
                total_loss = 0.0

        torch.save(model.state_dict(), f"chess_model_epoch")

    print("training complete!")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

npz_file = "data_path.npz"
dataset = ChessDataset(npz_file)

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = chess_model_v1.to(device)

train_model(model, train_loader, num_epochs=10, learning_rate=0.001, batch_size=64)

