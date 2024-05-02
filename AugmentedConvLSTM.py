import xarray as xr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json

class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ir = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ir = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hr = nn.Parameter(torch.Tensor(hidden_size))
        self.W_iz = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_iz = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hz = nn.Parameter(torch.Tensor(hidden_size))
        self.W_in = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_in = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hn = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(1), self.hidden_size, requires_grad=False)
        gates_r = input @ self.W_ir + hx @ self.W_hr + self.b_ir + self.b_hr
        gates_z = input @ self.W_iz + hx @ self.W_hz + self.b_iz + self.b_hz
        gates_n = input @ self.W_in + hx @ self.W_hn + self.b_in + self.b_hn
        reset_gate = torch.sigmoid(gates_r)
        update_gate = torch.sigmoid(gates_z)
        new_gate = torch.tanh(gates_n)
        hy = (1 - update_gate) * hx + update_gate * new_gate
        return hy


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_cell = CustomGRUCell(input_size, hidden_size)

    def forward(self, input, hx=None):
        seq_length, batch_size, _ = input.size()
        if hx is None:
            hx = input.new_zeros(batch_size, self.hidden_size, requires_grad=False)
        output = []
        for t in range(seq_length):
            hx = self.gru_cell(input[t], hx)
            output.append(hx.unsqueeze(0))
        output = torch.cat(output, dim=0)
        return output, hx


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = CustomGRU(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc1(gru_out[-1])
        output = torch.relu(output)
        output = self.fc2(output)
        return output


data = xr.open_dataset('myfile.nc')


df = data.to_dataframe().reset_index()


y = df['APCP_sfc'].values


X = df[['longitude', 'latitude']].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 1, 2)
X_test = X_test.reshape(-1, 1, 2)


input_size = 2
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 10
batch_size = 200


model = GRUModel(input_size, hidden_size, output_size)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(range(0, len(X_train), batch_size), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    for i in progress_bar:
        inputs = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
        labels = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)
        
       
        optimizer.zero_grad()

        
        outputs = model(inputs)

       
        loss = criterion(outputs.view(-1), labels)

        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        
        progress_bar.set_postfix({'loss': running_loss / ((i + batch_size) / batch_size)})

print('Finished Training')


def predict_rainfall(latitude, longitude):
    
    input_features = scaler.transform([[longitude, latitude]])

    
    input_tensor = torch.tensor(input_features, dtype=torch.float32).reshape(1, 1, 2)

    
    with torch.no_grad():
        output = model(input_tensor)

    
    predicted_rainfall = output.item()

    
    result = {
        "latitude": latitude,
        "longitude": longitude,
        "predicted_rainfall": predicted_rainfall
    }

    return result


latitude = 40.7128  
longitude = -74.0060  


result = predict_rainfall(latitude, longitude)


output_file = 'predicted_rainfall.json'
with open(output_file, 'w') as f:
    json.dump(result, f)

print(f"Predicted rainfall for latitude {latitude} and longitude {longitude}: {result['predicted_rainfall']} (mm)")
print(f"Result saved to {output_file}")
