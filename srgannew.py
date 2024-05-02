import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize
from PIL import Image

# Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# Custom Dataset class for rainfall images
class RainfallDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [filename for filename in os.listdir(data_dir) if filename.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_name)

        # Load image and preprocess
        img = Image.open(image_path).convert('RGB')  # Ensure RGB conversion
        img = Resize(IMAGE_SIZE)(img)  # Resize image
        img = ToTensor()(img)  # Convert to tensor
        img = img.float()  # Convert to float

        # Extract target (rainfall value) from filename
        target = self.get_rainfall_from_filename(image_name)

        return img, target

    def get_rainfall_from_filename(self, filename):
        # Example: 'rainfall_day_1.png' -> extract '1'
        return float(filename.split('_')[2].split('.')[0])

# U-Net Model Definition
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder (downsampling path)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Decoder (upsampling path)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)
        # Decoder
        x2 = self.decoder(x1)
        return x2

# Load data and create DataLoader
data_dir = 'rainfall'
dataset = RainfallDataset(data_dir)
train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Initialize U-Net model, loss function, and optimizer
model = UNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets.unsqueeze(1))  # targets need to be unsqueezed to match output shape
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            val_loss = criterion(outputs, targets.unsqueeze(1))
            val_losses.append(val_loss.item())

    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Validation Loss: {avg_val_loss:.4f}")


