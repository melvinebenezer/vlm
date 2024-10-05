import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X = torch.randn(1000, 10).to(device)
y = torch.randn(1000, 1).to(device)

model = SimpleNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epochs=10):
    for epoch in range(epochs):
        with record_function("training_epoch"):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        if epoch % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def inference():
    with record_function("inference"):
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(100, 10).to(device)
            predictions = model(test_input)
    print("Inference done. Shape of predictions:", predictions.shape)

def view_traced_model(traced_model):
    print("\n1. Printing the traced model:")
    print(traced_model)

    print("\n2. Viewing the graph attribute:")
    print(traced_model.graph)

    print("\n3. Using graph_for method:")
    print(traced_model.graph_for(torch.randn(1, 10).to(device)))

    print("\n4. Saving the graph to a file:")
    torch.onnx.export(traced_model, torch.randn(1, 10).to(device), "traced_model.onnx")
    print("Graph saved as 'traced_model.onnx'")

if __name__ == "__main__":
    print("Starting profiling...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True,
                 with_stack=True,
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
        print("Starting training...")
        train()
        print("Training completed. Starting inference...")
        inference()
    
    print("Profiling completed. Trace file saved in './log' directory.")
    
    # JIT tracing
    print("Starting JIT tracing...")
    traced_model = torch.jit.trace(model, torch.randn(1, 10).to(device))
    print("JIT tracing completed.")
    
    # Save the traced model
    traced_model.save("traced_model.pt")
    print("Traced model saved as 'traced_model.pt'")

    # View the traced model
    view_traced_model(traced_model)
    
    print("Script completed.")