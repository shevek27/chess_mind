import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

class ChessDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.current_position = 0
        self.batch_cache = []
        self.cache_loaded = False 

    def load_next_batch(self):
        with open(self.file_path, "rb") as file:
            file.seek(self.current_position)

            try:
                print("loading data")
                batch = pickle.load(file)
                self.batch_cache = list(zip(batch["features"], batch["labels"]))
                self.current_position = file.tell() # update the current position
                self.cache_loaded = True

            except EOFError:
                print("end of file")
                self.batch_cache = []
                self.cache_loaded = False


    def __len__(self):
        return 10**10 # extremely large, only have it because DataLoader needs it
    
    def __getitem__(self, index):
        # load new batch if cache is empty
        if not self.cache_loaded or index >= len(self.batch_cache):
            self.load_next_batch()

        # get the sample at index % batch size to index continuously
        if len(self.batch_cache) == 0:
            raise IndexError("no more data to load")
        
        sample_index = index % len(self.batch_cache)
        feature, label = self.batch_cache[sample_index]

        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)



        

# testing

file_path = "processed_lichess_2020_oct_filtered.pkl"
chess_dataset = ChessDataset(file_path)

data_loader = DataLoader(chess_dataset, batch_size=64, shuffle=False)

for features, labels in data_loader:
    print("batch loaded:")
    print(features.shape, labels.shape)
    print(features, labels)
    break


