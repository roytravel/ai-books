import torch

# input shape: batch_size(2) x feature dimension(3)
input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
m = torch.nn.LayerNorm(input.shape[-1])
output = m(input)
print (output)

