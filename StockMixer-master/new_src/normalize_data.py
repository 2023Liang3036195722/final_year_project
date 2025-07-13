import numpy as np

# 读取 .npy 文件
data = np.load('stock_data.npy')

# 查看数据的基本信息
print('数据类型：', type(data))
print('数据形状：', data.shape)

# 初始化归一化后的数据和记录参数的数组
normalized_data = np.zeros_like(data, dtype=np.float32)
max_values = np.zeros((data.shape[0], data.shape[2]))

# 对每个 (i, k) 组合的数据进行归一化
for i in range(data.shape[0]):
    for k in range(data.shape[2]):
        # 获取当前 (i, k) 组合的数据
        slice_data = data[i, :, k]
        # 计算最大值
        max_val = np.max(slice_data)
        normalized_data[i, :, k] = slice_data / max_val
        max_values[i, k] = max_val
# 保存归一化后的数据和参数
np.save('stock_data_normalized.npy', normalized_data)
np.save('stock_data_max_values.npy', max_values)
print("归一化完成并保存结果！")