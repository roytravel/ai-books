import torch
model = torch.nn.Dropout(p=0.2)
input = torch.randn(1, 10)
print (input)
output = model(input)
print (output)