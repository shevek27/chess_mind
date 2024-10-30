import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from chess_model_v1 import *

class ChessDataset(Dataset):
    def __init__(self, file_path, batch_size = 64):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.current_position = 0
        self.batch_cache = []
        self.batch_size = batch_size
        self.end_of_file = False

    def load_next_batch(self):

        if self.end_of_file:
            return

        with open(self.file_path, "rb") as file:
            file.seek(self.current_position)

            try:
                print("loading data...")
                batch = pickle.load(file)
                self.batch_cache = list(zip(batch["features"], batch["labels"]))
                self.current_position = file.tell() # update the current position

            except EOFError:
                print("end of file reached")
                self.batch_cache = []
                self.end_of_file = True

            print(f"loaded {len(self.batch_cache)} samples")


    def __len__(self):
        return 100000 # extremely large, only have it because DataLoader needs it
    
    def __getitem__(self, index):
        # load new batch if cache is empty
        if len(self.batch_cache) == 0:
            if self.end_of_file:
                raise IndexError("no more data to load")
            else:
                self.load_next_batch()
 
        if len(self.batch_cache) == 0:
            raise IndexError("batch cache data exhausted with no new data")
        
        sample_index = index % len(self.batch_cache)
        feature, label = self.batch_cache[sample_index]

        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def train_model(model, dataloader, num_epochs=10, learning_rate=0.001, batch_size=64):

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
            total_loss += loss

            loss.backward()

            optimizer.step()

            if batch_index % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], batch [{batch_index}], loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f"epoch: [{epoch+1} / {num_epochs}], loss: {avg_loss:.4f}")

first_model = chess_model_v1()

file_path = "processed_lichess_2020_oct_filtered.pkl"
chess_dataset = ChessDataset(file_path)
data_loader = DataLoader(chess_dataset, batch_size=256, shuffle=False)

train_model(first_model, data_loader, num_epochs=10, learning_rate=0.001, batch_size=256)

