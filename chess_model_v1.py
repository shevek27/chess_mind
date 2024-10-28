import torch
import torch.nn as nn
import torch.nn.functional as func

class chess_model_v1(nn.Module):
    def __init__(self):
        super(chess_model_v1).__init__()


        # convolutional layers

        self.conv1 = nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # fully connected layers

        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # pass data through conv layers with relu and max pooling
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.conv3(x))
        x = func.max_pool2d(x, 2)

        # flatten tensor for fully connected layers
        x = x.view(-1, 128 * 8 * 8)

        # pass through fully connected layers
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))

        # output layer with tanh to predict values -1, 0, 1
        x = torch.tanh(self.fc3(x))

        return x