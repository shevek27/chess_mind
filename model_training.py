import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
from chess_model_v1 import *

class ChessDataset(Dataset):
    def __init__(self, file_prefix):
        self.file_list = sorted(glob.glob(f"{file_prefix}_*.npz"))
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
    

def train_model(model, dataloader, save_path, num_epochs=100, learning_rate=0.001,  batch_size=64):

    model.train()

    loss_fn = nn.MSELoss() # mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        
        total_loss = 0

        for batch_index, (features, labels) in enumerate(dataloader):

            #labels = labels.view(-1,1)


            if len(features) == 0:
                print("no more data to train")
                break

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

        avg_loss = total_loss / len(dataloader)
        print(f"epoch: [{epoch+1} / {num_epochs}], loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), save_path)
        print(f"model saved to {save_path}")

save_path = "model_parameters"
model = chess_model_v1()

if os.path.isfile(save_path):
    print(f"loading model from {save_path}")
    model.load_state_dict(torch.load(save_path))

else:
    print(f"no model found at {save_path}. starting from zero")

file_prefix = "processed_dataset"
chess_dataset = ChessDataset(file_prefix)
data_loader = DataLoader(chess_dataset, batch_size=256, shuffle=False)

train_model(model, data_loader, save_path="chess_model_v1.pth", num_epochs=1, learning_rate=0.001, batch_size=256)

