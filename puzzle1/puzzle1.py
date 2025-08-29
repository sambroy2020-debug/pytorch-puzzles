import torch
import torch.nn as nn
import torch.optim as optim

class DynamicNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        # Buggy part: Using a standard Python list
        self.layers = []
        
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(nn.ReLU())
            
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

# Setup
model = DynamicNet(input_size=20, output_size=2, hidden_sizes=[32, 64])
optimizer = optim.SGD(model.parameters(), lr=0.01)
input_tensor = torch.randn(10, 20)
target = torch.empty(10, dtype=torch.long).random_(2)
loss_fn = nn.CrossEntropyLoss()

# Get initial weights of a "hidden" layer
initial_hidden_weights = model.layers[0].weight.clone()

# Training step
optimizer.zero_grad()
output = model(input_tensor)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()

# Check if the hidden layer weights have updated
print("Hidden layer weights updated:", not torch.equal(initial_hidden_weights, model.layers[0].weight))
print("Number of parameters found by optimizer:", len(list(model.parameters())))