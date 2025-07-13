import numpy as np
import torch as torch
from model import StockMixer


# # 预测计算量不大，改用cpu，默认就是cpu
# device = torch.device("cpu")
# input 类型:numpy.ndarray 形状: (83, 16, 5)
# 83只股票, 16个时间, 5个特征
# 股票顺序严格按照csv文件
# 时间连续，从小到大
# 5个特征，依次：'High' 'Low' 'Open' 'Volume' 'Close'

def predict(input_data):
    max_values = np.load('stock_data_max_values.npy')
    # 标准化
    normalized_data = np.zeros_like(input_data, dtype=np.float32)
    for i in range(input_data.shape[0]):
        for k in range(input_data.shape[2]):
            normalized_data[i, :, k] = input_data[i, :, k] / max_values[i, k]

    model = StockMixer(
        stocks=83,
        time_steps=16,
        channels=5,
        market=20,
        scale=3
    )
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    # input_tensor (83,16,5)
    input_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    with torch.no_grad():
        # (83,1)
        output = model(input_tensor)
        # 反标准化
        output = output.reshape(-1).numpy()
        denormalized_output = output * max_values[:, -1]
    return denormalized_output

if __name__ == '__main__':
    data = np.load('stock_data.npy')  # 加载原始数据
    timesteps = data[:, 10:26, :]
    print(timesteps.shape)
    x = data[:, 25, -1]
    y = data[:, 26, -1]
    predict_result = predict(timesteps)
    predict_ratio = (predict_result - x) / x * 100
    true_ratio = (y - x) / x * 100
    print('predict_price:', predict_result)
    print('true_price:', x)
    print('predict_ratio:', predict_ratio)
    print('true_ratio:', true_ratio)








