import math
import numpy as np
import torch as torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from model import StockMixer
from dataset import HKStockDataset

np.random.seed(123456789)
torch.random.manual_seed(12345678)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.")
else:
    print("CUDA not available!")
    exit()

# 超参数探索实验设置
# 7f5f63654c990eb6c7d796f534bf9db8cfcae73e
# 学习率
lr = 1e-3
hyper_m = 10
import wandb
wandb.login()
run = wandb.init(
    project="stock",
    name = "stock_200",
    config={
        "lr": lr,
        "hyper_m": hyper_m
    },
)

# 基本信息
stock_num = 83
windows_length = 16
fea_num = 5
market_num = hyper_m
scale_factor = 3


data = np.load('stock_data.npy')
dataset = HKStockDataset(data)

train_size = int(0.90 * len(dataset))
print("train_size:", train_size)
test_size = len(dataset) - train_size
print("test_size:", test_size)

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
optimizer = optim.Adam(model.parameters(), lr=lr)

print('begin training...')
total_train_batches = len(train_dataloader)
total_test_batches = len(test_dataloader)
print('total train batches'+str(total_train_batches))
print('total test_batches'+str(total_test_batches))

model.train()
num_epochs = 200
for epoch in range(num_epochs):
    running_loss = 0.0

    # batch_idx = 0
    for inputs, labels in train_dataloader:
        # batch_idx += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # if batch_idx % 100 == 0:
        #     print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{total_batches}, Loss: {loss.item():.6f}')

    # 计算训练平均损失开方
    avg_sqrt_train_loss = math.sqrt(running_loss / total_train_batches)

    # test
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    model.train()

    # 计算平均测试损失
    avg_sqrt_test_loss = math.sqrt(test_loss / total_test_batches)

    print(f'Epoch {epoch + 1}/{num_epochs}, train_sqrt_loss: {avg_sqrt_train_loss:.6f},'
          f' test_sqrt_loss: {avg_sqrt_test_loss:.6f}')
    wandb.log({"train_sqrt_loss": avg_sqrt_train_loss, "test_sqrt_loss": avg_sqrt_test_loss})
