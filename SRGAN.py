import xarray as xr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class CustomESN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, alpha=1.0, spectral_radius=0.9):
        super(CustomESN, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.alpha = alpha
        self.spectral_radius = spectral_radius

        # Reservoir weights
        self.W_reservoir = nn.Parameter(torch.empty(reservoir_size, reservoir_size).uniform_(-1, 1))

        # Input-to-reservoir weights
        self.W_in = nn.Parameter(torch.empty(input_size, reservoir_size).uniform_(-1, 1))

        # Reservoir-to-output weights
        self.W_out = nn.Parameter(torch.empty(reservoir_size, output_size).uniform_(-1, 1))

        # Initialize the weights
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize reservoir weights with uniform distribution
        nn.init.uniform_(self.W_reservoir, -1, 1)

        # Initialize input-to-reservoir weights with uniform distribution
        nn.init.uniform_(self.W_in, -1, 1)

        # Initialize reservoir-to-output weights with uniform distribution
        nn.init.uniform_(self.W_out, -1, 1)

        # Scale the spectral radius of the reservoir weights
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(self.W_reservoir)
            radius = torch.max(torch.abs(eigenvalues))
            self.W_reservoir /= radius / self.spectral_radius

    def forward(self, input):
        batch_size = input.size(0)
        seq_length = input.size(1)

        # Initialize reservoir states
        reservoir_state = torch.zeros(batch_size, self.reservoir_size, dtype=torch.float32)

        # Iterate over time steps
        for t in range(seq_length):
            # Input at time t
            x = input[:, t, :]

            # Update reservoir state
            reservoir_state = (1 - self.alpha) * reservoir_state + \
                              self.alpha * torch.tanh(reservoir_state @ self.W_reservoir + x @ self.W_in)

        # Output
        output = reservoir_state @ self.W_out

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

# Reshape input features for ESN
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define hyperparameters
input_size = 2
reservoir_size = 100
output_size = 1
alpha = 0.5
spectral_radius = 0.9
learning_rate = 0.001
num_epochs = 10
batch_size = 200

# Create the ESN model
model = CustomESN(input_size, reservoir_size, output_size, alpha, spectral_radius)

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
