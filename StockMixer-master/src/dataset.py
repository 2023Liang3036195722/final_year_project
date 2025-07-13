import torch
from torch.utils.data import Dataset

# windows_length 信息窗口 仅窗口内的数据对模型可见
# data N只股票 * T个时间 F个特征
class HKStockDataset(Dataset):
    def __init__(self, data, steps=1, windows_length=16):
        self.data = data
        self.steps = steps
        self.windows_length = windows_length

    def __len__(self):
        return (self.data.shape[1] - self.windows_length - 1) // self.steps + 1

    def __getitem__(self, idx):
        i = idx * self.steps
        data_input = self.data[:, i:i + self.windows_length, :]
        x = self.data[:, i + self.windows_length-1, -1]
        y = self.data[:, i + self.windows_length, -1]
        data_label = (y-x)/x  * 100 # 因为标签太小，增大一下

        # 转换数据类型为 torch.Tensor
        data_input = torch.tensor(data_input, dtype=torch.float32)
        data_label = torch.tensor(data_label, dtype=torch.float32)

        return data_input, data_label

# windows_length 信息窗口 仅窗口内的数据对模型可见
# data N只股票 * T个时间 F个特征
class HKStockDataset_prices(Dataset):
    def __init__(self, data, steps=1, windows_length=16):
        self.data = data
        self.steps = steps
        self.windows_length = windows_length

    def __len__(self):
        return (self.data.shape[1] - self.windows_length - 1) // self.steps + 1

    def __getitem__(self, idx):
        i = idx * self.steps
        data_input = self.data[:, i:i + self.windows_length, :]
        x = self.data[:, i + self.windows_length-1, -1]
        y = self.data[:, i + self.windows_length, -1]
        data_ratio = (y-x)/x  * 100 # 因为标签太小，增大一下

        # 转换数据类型为 torch.Tensor
        data_input = torch.tensor(data_input, dtype=torch.float32)
        data_ratio = torch.tensor(data_ratio, dtype=torch.float32)
        data_base = torch.tensor(x, dtype=torch.float32)

        return data_input, data_ratio, data_base