import numpy as np
import torch as torch
from model import StockMixer
from sklearn.metrics import mean_absolute_error, mean_squared_error


# # 预测计算量不大，改用cpu，默认就是cpu
# device = torch.device("cpu")

# input 类型:numpy.ndarray 形状: (83, 16, 5)
# 83只股票, 16个时间, 5个特征
# 'High'
# 'Low'
# 'Open'
# 'Volume'
# 'Close'
def predict(input_data):
    model = StockMixer(
        stocks=83,
        time_steps=16,
        channels=5,
        market=10,
        scale=3
    )
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    # input_tensor (83,16,5)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        # (83,1)
        output = model(input_tensor)
    # (83)
    return output.reshape(-1).numpy()

if __name__ == '__main__':
    data = np.load('stock_data_normalized.npy')
    timesteps = data[:, 10:26, :]
    print(timesteps.shape)
    x = data[:, 25, -1]
    y = data[:, 26, -1]
    predict_result = predict(timesteps)
    predict_ratio = (predict_result-x)/x*100
    true_ratio = (y-x)/x*100
    print('predict_price:',predict_result)
    print('true_price:',x)
    print('predict_ratio:',predict_ratio)
    print('true_ratio:',true_ratio)








