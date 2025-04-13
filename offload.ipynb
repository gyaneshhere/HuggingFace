import torch
import torch.nn as nn

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = MyModel()

# Check if CUDA (GPU) is available and move the model to the GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a sample input tensor
input_tensor = torch.randn(1, 10).to(device)

# Perform a forward pass on the GPU
output_tensor_gpu = model(input_tensor)
print(f"Output on GPU: {output_tensor_gpu}")

# Offload the model parameters to the CPU
model.to("cpu")

# Create a sample input tensor on the CPU
input_tensor_cpu = torch.randn(1, 10)

# Perform a forward pass on the CPU
output_tensor_cpu = model(input_tensor_cpu)
print(f"Output on CPU: {output_tensor_cpu}")
