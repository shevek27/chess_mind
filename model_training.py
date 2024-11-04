import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
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
    def __init__(self, file_prefix, specific_file):
        if not specific_file:
            self.file_list = sorted(glob.glob(f"{file_prefix}_*.npz"))
        else:
            self.file_list = [specific_file]
        self.file_sizes = []
        self.total_samples = 0

        for file_path in self.file_list:
            data = np.load(file_path)
            num_samples = len(data['labels'])
            self.file_sizes.append(num_samples)
            self.total_samples += num_samples

        self.current_file_index = 0
        self.current_data = None
        self.load_file(self.current_file_index)


    def reset(self):
        # resets dataset at the beginning of each epoch
        self.current_file_index = 0
        self.features = None
        self.labels = None
        self.load_file(self.current_file_index)


    def load_file(self, file_index):
        # load file given the index
        file_path = self.file_list[file_index]
        print(f"loading data from {file_path}")
        loaded_data = np.load(file_path)
        self.features = loaded_data["features"]
        self.labels = loaded_data["labels"]
        self.current_file_size = len(self.labels) # number of samples



    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx >= self.total_samples:
            raise IndexError("index out of range")
        
        
        # determine the file and local index inside that file
        file_index = 0
        total_size = 0
        for size in self.file_sizes:
            if idx < total_size + size:
                break
            total_size += size
            file_index += 1



        if file_index != self.current_file_index:
            self.current_file_index = file_index
            self.load_file(self.current_file_index)

        in_file_index = idx - total_size

        feature = self.features[in_file_index]
        label = self.labels[in_file_index]

        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    

    

def train_model(model, train_loader, val_loader, save_path, num_epochs=100, learning_rate=0.001,  batch_size=64):

    model.train()
    loss_fn = nn.MSELoss() # mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=3, delta=0, save_path=save_path)

    for epoch in range(num_epochs):
 
        model.train()
        total_loss = 0

        for batch_index, (features, labels) in enumerate(train_loader):
            if batch_index >= 1000:
                break
            if len(features) == 0:
                print("no more data to train")
                break

            features = features.to(device)
            labels = labels.to(device)

            # reshape input tensor to (batch_size, channels, height, width)
            features = features.permute(0,3,1,2)

            optimizer.zero_grad()

            outputs = model(features) 

            loss = loss_fn(outputs.view(-1), labels)
            total_loss += loss.item()

            loss.backward()

            optimizer.step()

            if batch_index % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], batch [{batch_index}], loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f"epoch: [{epoch+1} / {num_epochs}], loss: {avg_loss:.4f}")
        print(f"model saved to {save_path}")

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for features, labels in val_loader:
                features = features.permute(0,3,1,2).to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = loss_fn(outputs.view(-1), labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("early stopping triggered, exiting training loop")
            break

        model.train()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

save_path = "chess_model_v1.pth"
model = chess_model_v1().to(device)

if os.path.isfile(save_path):
    print(f"loading model from {save_path}")
    model.load_state_dict(torch.load((save_path), map_location=device))

else:
    print(f"no model found at {save_path}. starting from zero")

file_prefix = "processed_dataset"
train_dataset = ChessDataset(file_prefix, specific_file=False)
val_dataset = ChessDataset(file_prefix= False, specific_file="val_dataset.npz")
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False)

train_model(model, train_loader, val_loader, save_path="chess_model_v1.pth", num_epochs=50, learning_rate=0.001, batch_size=256)

