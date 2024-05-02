import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Define U-Net architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
            DoubleConv(512, 1024)
        ])
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(128, 128),  # Update to match encoder output channels
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(64, 64)     # Update to match encoder output channels
        ])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2)(x)

        # Reverse skip connections to match decoder order
        skip_connections.reverse()

        # Decoder
        for i, decoder in enumerate(self.decoder):
            if i < len(skip_connections):
                x = torch.cat((x, skip_connections[i]), dim=1)  # Concatenate along channel dimension
                x = decoder(x)
            else:
                x = decoder(x)  # Apply decoder without skip connection for the last decoder layer

        # Final convolution
        x = self.final_conv(x)
        return x



# Define custom dataset class
class RainfallDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name)
        # Apply any necessary preprocessing transformations
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a common size
    transforms.ToTensor()            # Convert images to tensors
])

# Load dataset and create DataLoader
batch_size = 16  # Set batch size to match model's expected input size
dataset = RainfallDataset(root_dir='rainfall', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize U-Net model
in_channels = 4  # Update input channels to match dataset
out_channels = 1
model = UNet(in_channels, out_channels)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define custom gradient loss function
def gradient_loss(pred, target):
    pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    loss = torch.mean(torch.abs(pred_grad_x - target_grad_x) + torch.abs(pred_grad_y - target_grad_y))
    return loss

# Define number of epochs
num_epochs = 10

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, inputs in enumerate(dataloader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs)
        targets = inputs.clone().detach().to(device)  # Assuming ground truth is available in the dataset
        loss = gradient_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')
