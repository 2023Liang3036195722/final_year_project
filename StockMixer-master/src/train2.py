import numpy as np
import torch as torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from model2 import StockMixer
from dataset import HKStockDataset

np.random.seed(123456789)
torch.random.manual_seed(12345678)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.")
else:
    print("CUDA not available!")
    exit()


stock_num = 10
windows_length = 16
fea_num = 5
market_num = 2
scale_factor = 3


data = np.load('../stock_data.npy')
dataset = HKStockDataset(data)

train_size = int(0.95 * len(dataset))
test_size = len(dataset) - train_size

train_dataset = Subset(dataset, range(train_size))
test_dataset = Subset(dataset, range(train_size, len(dataset)))

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = StockMixer(
    stocks=stock_num,
    time_steps=windows_length,
    features=fea_num,
    market=market_num,
    scale=scale_factor
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('begin training...')

total_batches = len(train_dataloader)
print('total batches'+str(total_batches))

model.train()
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0

    batch_idx = 0
    for inputs, labels in train_dataloader:
        batch_idx += 1

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{total_batches}, Loss: {loss.item():.6f}')
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / total_batches:.6f}')

    # model.eval()
    # test_loss = 0.0
    # total_test_batches = len(test_dataloader)
    # with torch.no_grad():
    #     for inputs, labels in test_dataloader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         test_loss += loss.item()
    # print(f'Test Loss: {test_loss / total_test_batches:.6f}')
    # model.train()


print('begin testing...')
model.eval()
test_loss = 0.0
total_test_batches = len(test_dataloader)
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
print(f'Test Loss: {test_loss / total_test_batches:.6f}')
