# %% load library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# %% define model architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 64) # 4608 is basically 12 X 12 X 32
        self.fc2 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        op = F.log_softmax(x, dim=1)
        return op

# %% define train and test function 
def train(model, device, train_dataloader, optim, epoch):
    model.train() # change mode to train.
    for batch_idx, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device) # 
        optim.zero_grad() # do not accumulate gradient what previously used. 
        pred_prob = model(X) # get prediction probability on input data X
        loss = F.nll_loss(pred_prob, y) # the negative log likelihood loss.
        loss.backward()
        optim.step()
        if batch_idx % 10 == 0:
            print ("epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}".format\
                (epoch, batch_idx * len(X), len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader), loss.item()))


def test(model, device, test_dataloader):
    model.eval() # change mode to evaluation.
    loss = 0
    success = 0
    with torch.no_grad(): # for inference
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)

            # sum on loss of each mini batch
            loss = loss + F.nll_loss(pred_prob, y, reduction='sum').item() # item plays role of getting value of tensor
            pred = pred_prob.argmax(dim=1, keepdim=True) # get maximum 
            success = success + pred.eq(y.view_as(pred)).sum().item() # eq is computing element-wise equality

    loss = loss / len(test_dataloader.dataset) # divide loss by length of test dataset
    print ('\nTest dataset: Overall Loss: {:.4f}, Overall Accuracy: {}/{} ({:.0f}%)\n'.format(loss, success, len(test_dataloader.dataset), 100. * success / len(test_dataloader.dataset)))


# %% loading dataset
train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1302,), (0.3069,))])), # train_X.mean()/256. and train_X.std()/256.
    batch_size=32, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1302,), (0.3069,)) 
        ])),
    batch_size=500, shuffle=False)

# %% set hyper parameter
torch.manual_seed(0)
device = torch.device("cpu")
model = ConvNet()
optimizer = optim.Adadelta(model.parameters(), lr=0.5)

# %% do train and test
for epoch in range(1, 3):
    train(model, device, train_dataloader, optimizer, epoch)
    test(model, device, test_dataloader)

# %% plot sample data
test_samples = enumerate(test_dataloader)
batch_idx, (sample_data, sample_targets) = next(test_samples)

plt.imshow(sample_data[0][0], cmap='gray', interpolation='none')

# %% comprison between model prediction and ground truth.
print(f"Model prediction is : {model(sample_data).data.max(1)[1][0]}")
print(f"Ground truth is : {sample_targets[0]}")

# %% save model
torch.save(model.state_dict(), "./mnist.pt")
