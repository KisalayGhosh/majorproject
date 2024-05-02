import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# Define the AttentionBlock
class AttentionBlock(nn.Module):
    def __init__(self, hidden_channels):
        super(AttentionBlock, self).__init__()
        self.hidden_channels = hidden_channels
        self.query_conv = nn.Conv2d(hidden_channels, hidden_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(hidden_channels, hidden_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, hidden_channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.hidden_channels, height, width)
        out = self.gamma * out + x
        return out

# Define the ConvLSTMCell with Attention
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv_i = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.conv_f = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.conv_o = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.conv_c = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)

        self.attention = AttentionBlock(hidden_channels)

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state
        combined = torch.cat((x, h_prev), dim=1)

        i = torch.sigmoid(self.conv_i(combined))
        f = torch.sigmoid(self.conv_f(combined))
        o = torch.sigmoid(self.conv_o(combined))
        g = torch.tanh(self.conv_c(combined))

        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)

        h_attn = self.attention(h_new)
        return h_attn, c_new

# Define the ACLSTM model
class ACLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ACLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv_lstm_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, x):
        batch_size, seq_length, input_channels, height, width = x.size()
        h, c = self.init_hidden(batch_size, height, width)

        outputs = []
        for t in range(seq_length):
            h, c = self.conv_lstm_cell(x[:, t, :, :, :], (h, c))
            outputs.append(h.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def init_hidden(self, batch_size, height, width):
         device = next(self.parameters()).device
         return (torch.zeros(batch_size, self.hidden_channels, height, width).to(device),
         torch.zeros(batch_size, self.hidden_channels, height, width).to(device))


# Define custom dataset class for NetCDF data
class RainfallDataset(Dataset):
    def __init__(self, nc_file_path, sequence_length, transform=None):
        self.nc_file_path = nc_file_path
        self.sequence_length = sequence_length
        self.transform = transform
        self.file_list = []

        # Open the NetCDF file
        try:
            self.nc_file = nc.Dataset(nc_file_path, 'r')
            self.image_data = self.nc_file.variables['RAINFALL']
            num_time_steps = self.image_data.shape[0]

            # Generate list of file names for sequences
            for i in range(num_time_steps - sequence_length + 1):
                self.file_list.append((i, f'rainfall_sequence_{i + 1}.png'))

        except Exception as e:
            print(f"Error opening NetCDF file: {e}")
            exit(1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        idx_sequence, filename = self.file_list[idx]
        sequence_images = []

        # Load images for the sequence
        for j in range(self.sequence_length):
            image_array = self.image_data[idx_sequence + j, :, :]
            sequence_images.append(image_array)

        # Convert list of images to numpy array
        sequence_images = np.array(sequence_images)

        # Plot the sequence of images
        fig, axes = plt.subplots(1, self.sequence_length, figsize=(4 * self.sequence_length, 4))
        for j, image_array in enumerate(sequence_images):
            axes[j].imshow(image_array, cmap='viridis')
            axes[j].set_title(f'Day {idx_sequence + j + 1}')
            axes[j].axis('off')
        plt.tight_layout()

        # Save the plot as an image file (PNG format)
        output_dir = 'rainfallsequence'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, filename)
        plt.savefig(output_file)
        plt.close()

        # Apply transformations if provided
        if self.transform:
            sequence_images = [self.transform(image) for image in sequence_images]

        return sequence_images

# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.ToTensor()  # Convert images to tensors
])

# Specify the path to the NetCDF file
nc_file_path = 'rainfall.nc'

# Create the dataset
sequence_length = 5
dataset = RainfallDataset(nc_file_path, sequence_length, transform=transform)

# Load dataset using DataLoader
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize ACLSTM model
input_channels = 1  # Assuming grayscale images
hidden_channels = 64
kernel_size = 3
aclstm_model = ACLSTM(input_channels, hidden_channels, kernel_size)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aclstm_model.to(device)

# Define optimizer
optimizer = optim.Adam(aclstm_model.parameters(), lr=0.001)

# Define number of epochs
num_epochs = 10

# Train the model
for epoch in range(num_epochs):
    aclstm_model.train()
    running_loss = 0.0
    for batch_idx, inputs_list in enumerate(dataloader):
        optimizer.zero_grad()

        # Convert list of images to a tensor and ensure the correct data type
        inputs = torch.stack(inputs_list, dim=1).to(device).float()  # Convert to float data type

        outputs = aclstm_model(inputs)
        targets = inputs.clone().detach().to(device)  # Assuming ground truth is available
        loss = torch.nn.functional.mse_loss(outputs, targets)  # Using mean squared error loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')


