import xarray as xr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class PeepholeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PeepholeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Peephole connections weights
        self.W_ci = nn.Parameter(torch.empty(hidden_size))
        self.W_cf = nn.Parameter(torch.empty(hidden_size))
        self.W_co = nn.Parameter(torch.empty(hidden_size))

        # Forget gate weights
        self.W_if = nn.Parameter(torch.empty(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Input gate weights
        self.W_ii = nn.Parameter(torch.empty(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Cell gate weights
        self.W_ig = nn.Parameter(torch.empty(input_size, hidden_size))
        self.W_hg = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Output gate weights
        self.W_io = nn.Parameter(torch.empty(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        nn.init.uniform_(self.W_ci, -std, std)
        nn.init.uniform_(self.W_cf, -std, std)
        nn.init.uniform_(self.W_co, -std, std)
        nn.init.uniform_(self.W_if, -std, std)
        nn.init.uniform_(self.W_hf, -std, std)
        nn.init.uniform_(self.W_ii, -std, std)
        nn.init.uniform_(self.W_hi, -std, std)
        nn.init.uniform_(self.W_ig, -std, std)
        nn.init.uniform_(self.W_hg, -std, std)
        nn.init.uniform_(self.W_io, -std, std)
        nn.init.uniform_(self.W_ho, -std, std)

    def forward(self, input, hidden):
        h_prev, c_prev = hidden

        # Forget gate
        forget_gate = torch.sigmoid(input @ self.W_if + h_prev @ self.W_hf + c_prev * self.W_cf)

        # Input gate
        input_gate = torch.sigmoid(input @ self.W_ii + h_prev @ self.W_hi + c_prev * self.W_ci)

        # Cell gate
        cell_gate = torch.tanh(input @ self.W_ig + h_prev @ self.W_hg)

        # Update cell state
        cell_state = forget_gate * c_prev + input_gate * cell_gate

        # Output gate
        output_gate = torch.sigmoid(input @ self.W_io + h_prev @ self.W_ho + cell_state * self.W_co)

        # Update hidden state
        hidden_state = output_gate * torch.tanh(cell_state)

        return hidden_state, cell_state

class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PeepholeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = PeepholeLSTMCell(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)
        c = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)

        # Iterate over time steps
        for t in range(seq_length):
            h, c = self.lstm_cell(x[:, t, :], (h, c))

        # Output
        output = self.fc1(h)
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
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define hyperparameters
input_size = 2
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 10
batch_size = 200

# Create the Peephole LSTM model
model = PeepholeLSTM(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with tqdm
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(range(0, len(X_train), batch_size), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    for i in progress_bar:
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

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
