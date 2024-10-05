import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create random data
X = torch.randn(1000, 10).to(device)
y = torch.randn(1000, 1).to(device)

# Initialize the model and move it to GPU if available
model = SimpleNet().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
def train(epochs=10000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Inference
def inference():
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(100, 10).to(device)
        predictions = model(test_input)
    print("Inference done. Shape of predictions:", predictions.shape)

if __name__ == "__main__":
    print("Starting training...")
    train()
    print("Training completed. Starting inference...")
    inference()
    print("Script completed.")