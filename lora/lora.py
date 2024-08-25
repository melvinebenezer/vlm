import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

_ = torch.manual_seed(0)

# Network to classify MNIST digits and fine tune the network

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081,))])

# load the MINIST dataset
mnist_trainset = datasets.MINST(root='./data', train=True, download=True, transform=transform)

# Create a dataloader for the training
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, hidden_size_1=1000, hidden_size_2=2000 ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)

# Let's train the network for only 1 epoch

def train(train_loader, net, epochs=5, total_iterations_limit=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    total_iterations = 0

    for epoch in range(epochs):
        net.train()

        loss_sum = 0
        num_iterations = 0

        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = net(x.view(-1, 28*28))
            loss = criterion(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return 

train(train_loader, net, epochs-1)

#keep a copy of the original weights
original_weights = {}
for name, param in net.named_parameters():
    original_weights[name] = param.clone().detach()


def test(test_loader, net):
    correct = 0
    total = 0
    wrong_counts = [0 for i in range(10)]

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = net(x.view(-1, 28 * 28))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                else:
                    wrong_counts[y[idx]] += 1
    print(f"Accuracy: {round(correct / total, 3)}")
    for i in range(len(wrong_counts)):
        print(f"Wrong counts for the digit {i}: {wrong_counts[i]}")

test(test_loader, net)


# Visualise how many parameters are there in the network

total_parameters_original = 0
for index, layer in enumerate([net.fc1, net.fc2, net.fc3]):
    total_parameters_original += layer.weight.numel()
    total_parameters_original += layer.bias.numel()
    print(f"Number of parameters in layer {index + 1}: {layer.weight.numel() + layer.bias.numel()}")
print(f"Total number of parameters in the network: {total_parameters_original}")


# Define the LoRA parameterization as in the paper

class LoRAParameterization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
        super().__init__()
        # from section 4.1 of the paper
        # we use the random Gaussian initialization for A and zero for B.  W = AB is zero at the begining of the training
        self.lora_A = nn.Parameter(torch.zeros(rank, features_out)).to(device)
        self.lora_B = nn.Parameter(torch.zeros(features_in, rank)).to(device)
        nn.init.normal_(self.lora_A, mean=0, std=1)

        # Section 4.1 of the paper
        # We then scale Wx by a/r where a is a constant in r
        # when optimizing with adam tuning a is roughly the same as tuning the learning rate if we scale the initialization appropriately
        # As a result, we simple set it ti the firs r we try and do not tune it.abs
        # This scaling helps to reduce the need to retune the hyperparameters when we vary r
        self.alpha = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B * A) * scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights

# Add the parameterization to the network

import torch.nn.utils.parameterize as parameterize

def linear_layer_parameterization(layer, device, rank=1, alpha=1):
    # Only add the parameterization to the weight matrix, ignore the bias

    # from section 4.2 of the paper 
    # We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules ( so they are not trained on downstream tasks) bot 
    # [...]
    # We leave the empricial investigation of [...], and biases to a future work

    features_in, features_out = layer.weight.shape
    return LoRAParameterization(
        features_in, features_out, rank=rank, alpha=alpha, device=device
    )
    

parameterize.register_parameterization(
    net.fc1, "weight", linear_layer_parameterization(net.fc1, device)
)

parameterize.register_parameterization(
    net.fc2, "weight", linear_layer_parameterization(net.fc2, device)
)

parameterize.register_parameterization(
    net.fc3, "weight", linear_layer_parameterization(net.fc3, device)
)


def enable_disable_lora(enabled=True):
    for layer in [net.fc1, net.fc2, net.fc3]:
        layer.parameterization["weight"].enabled = enabled


# Visualize the parameters

total_parameters_lora = 0
total_parameters_non_lora = 0
for index, layer in enumerate([net.fc1, net.fc2, net.fc3]):
    total_parameters_lora += layer.parameterization["weight"].lora_A.numel()
    total_parameters_lora += layer.parameterization["weight"].lora_B.numel()
    total_parameters_non_lora += layer.weight.numel()
    total_parameters_non_lora += layer.bias.numel()
    print(f"Number of parameters in layer {index + 1}: {layer.parameterization['weight'].lora_A.numel() + layer.parameterization['weight'].lora_B.numel()}")

# The non-LoRA parameters count must match the original network
assert total_parameters_non_lora == total_parameters_original
print(f"Total number of parameters in the network: {total_parameters_non_lora}")
print(f"Total number of parameters in the LoRA parameterization: {total_parameters_lora}")
print(f"Total number of parameters in the network with Original + LoRA: {total_parameters_original + total_parameters_lora}")
parameters_increment = (total_parameters_lora / total_parameters_original) * 100    
print (f"Parameters increment: {parameters_increment}%")


# Freeze the non-LoRA parameters
for name, param in net.named_parameters():
    if 'lora' not in name:
        print(f"freezing non-lora parameter: {name}")
        param.requires_grad = False

# load the MNIST data set again
mnist_trainset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  
exclude_indices = mnist_trainset.targets == 9
mnist_trainset.data = mnist_trainset.data[exclude_indices]
mnist_trainset.targets = mnist_trainset.targets[exclude_indices]

train(train_loader, net, epochs=1, total_iterations_limit=100)


# Verify the fine tuning did not alter the original weights

assert torch.all(net.fc1.parameterizations.weight.original == original_weights["fc1"])
assert torch.all(net.fc2.parameterizations.weight.original == original_weights["fc2"])
assert torch.all(net.fc3.parameterizations.weight.original == original_weights["fc3"])

enable_disable_lora(enabled=True)
# The new linear1.weight is obtained by the forward function of our LoRA parameterization
# THe original weights have been moved to net.linear1.parameterizations.weight.original_weights

assert torch.equal(net.fc1.weight, net.fc1.parameterizations.weight.original + (net.fc1.parameterizations.weight.lora_B @ net.fc1.parameterizations.weight.lora_A).view(net.fc1.weight.shape) * net.fc1.parameterizations.weight.scale)

enable_disable_lora(enabled=False)
# If we disable LoRA, the linear1.weight is the original weight

assert torch.equal(net.fc1.weight, original_weights["fc1"])

# The accuracy of digit 9 would have improved
enable_disable_lora(enabled=True)
test(test_loader, net)

# should provide the original accuracy
enable_disable_lora(enabled=False)
test(test_loader, net) 

# Now we need to only save the loRA parameters
# save the LoRA parameters
torch.save(
    {
        "fc1": {
            "lora_A": net.fc1.parameterizations["weight"].lora_A,
            "lora_B": net.fc1.parameterizations["weight"].lora_B,
        },
        "fc2": {
            "lora_A": net.fc2.parameterizations["weight"].lora_A,
            "lora_B": net.fc2.parameterizations["weight"].lora_B,
        },
        "fc3": {
            "lora_A": net.fc3.parameterizations["weight"].lora_A,
            "lora_B": net.fc3.parameterizations["weight"].lora_B,
        },
    },
    "lora_parameters.pth",
)

# Re load the network and the LoRA parameters
net = Net().to(device)
checkpoint = torch.load("lora_parameters.pth")
net.fc1.parameterizations["weight"].lora_A = checkpoint["fc1"]["lora_A"]
net.fc1.parameterizations["weight"].lora_B = checkpoint["fc1"]["lora_B"]
net.fc2.parameterizations["weight"].lora_A = checkpoint["fc2"]["lora_A"]
net.fc2.parameterizations["weight"].lora_B = checkpoint["fc2"]["lora_B"]
net.fc3.parameterizations["weight"].lora_A = checkpoint["fc3"]["lora_A"]
net.fc3.parameterizations["weight"].lora_B = checkpoint["fc3"]["lora_B"]


