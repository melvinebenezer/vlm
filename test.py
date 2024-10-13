import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import urllib.request
import tarfile
from pathlib import Path
from torchao.prototype.dtypes.bitnet import BitnetTensor

## TODO:: @Z
# NotImplementedError: aten.amin.default <<

class BitLinearTrain(nn.Linear):
    def forward(self, x):
        w = self.weight
        x_norm = x
        x_quant = x_norm + (self.activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (self.weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y

    def activation_quant(self, x):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
        return y

    def weight_quant(self, w):
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
        return u

class ImageMLP(nn.Module):
    def __init__(self):
        super(ImageMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = BitLinearTrain(160 * 160 * 3, 8)
        self.softmax = nn.Softmax(dim=1)

    def replace_linears(self):
        new_linear = nn.Linear(self.linear.in_features, self.linear.out_features, device=self.linear.weight.device, bias=None)
        new_linear.weight = torch.nn.Parameter(BitnetTensor.from_float(self.linear.weight), requires_grad=False)
        self.linear = new_linear

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

dataset_path = "imagenette2-160"
if not Path(dataset_path).exists():
    # Set the URL and local path for the dataset
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    local_path = "imagenette2-160.tgz"

    # Download the dataset
    urllib.request.urlretrieve(url, local_path)

    # Extract the dataset
    with tarfile.open(local_path, "r:gz") as tar:
        tar.extractall()

    # Remove the downloaded archive
    os.remove(local_path)

# Define the transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset using ImageFolder
dataset = ImageFolder(root=dataset_path, transform=transform)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Print the sizes of the train and validation sets
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model = ImageMLP().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate loss and accuracy
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss = train_loss / len(train_dataset)
    train_acc = train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate loss and accuracy
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_dataset)
    val_acc = val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

model.replace_linears()

# Validation
model.eval()
val_loss = 0.0
val_correct = 0
val_total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Calculate loss and accuracy
        val_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

val_loss = val_loss / len(val_dataset)
val_acc = val_correct / val_total

print(f"L: {val_loss} A: {val_acc}")
