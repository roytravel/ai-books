import torch

x = torch.tensor([2, 1]) # input
w1 = torch.tensor([[3, 2, -4], [2, -3, 1]]) # link between input layer and hidden layer
b1 = 1
w2 = torch.tensor([[-1, 1], [1, 2], [3, 1]]) # link between hidden layer and output layer
b2 = -1

h_preact = torch.matmul(x, w1) + b1
h = torch.nn.functional.relu(h_preact)
y = torch.matmul(h, w2) + b2
print (h_preact)
print (h)
print  (y)
