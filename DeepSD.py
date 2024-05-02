import xarray as xr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(1), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        seq_length, batch_size, _ = input.size()
        hidden_seq = []
        for t in range(seq_length):
            hx = self.lstm_cell(input[t], hx)
            hidden_seq.append(hx[0].unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq, hx

    def lstm_cell(self, input, hx):
        h_prev, c_prev = hx
        gates = input @ self.W_ih + h_prev @ self.W_hh + self.b_ih + self.b_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = (forgetgate * c_prev) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = CustomLSTM(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc1(lstm_out[-1])
        output = torch.relu(output)
        output = self.fc2(output)
        return output

# Load the data from the .nc file
data = xr.open_dataset('myfile.nc')

# Convert data to pandas DataFrame
df = data.to_dataframe().reset_index()

# Extracting target variable (APCP_sfc)
y = df['APCP_sfc'].values

# Extracting input features (longitude and latitude)
X = df[['longitude', 'latitude']].values

# Standardize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input features for LSTM
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 1, 2)
X_test = X_test.reshape(-1, 1, 2)

# Define hyperparameters
input_size = 2
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 10
batch_size = 200

# Create the LSTM model
model = LSTMModel(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with tqdm
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(range(0, len(X_train), batch_size), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    for i in progress_bar:
        inputs = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
        labels = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs.view(-1), labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update tqdm progress bar description
        progress_bar.set_postfix({'loss': running_loss / ((i + batch_size) / batch_size)})

print('Finished Training')
