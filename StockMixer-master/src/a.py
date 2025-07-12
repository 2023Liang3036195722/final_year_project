import numpy as np

# # 读取 .npy 文件
# data = np.load('stock_data.npy')
#
# # 查看数据的基本信息
# print('数据类型：', type(data))
# print('数据形状：', data.shape)
# print('数据内容：')
# print(data)

# 读取 .npy 文件
data = np.load('stock_data.npy')

# 查看数据的基本信息
print('数据类型：', type(data))
print('数据形状：', data.shape)
print('数据内容：')
print(data)

# 获取第二个维度（索引为1）的最后16个数据
last_16_timesteps = data[:, -16:, :]  # 形状为 (batch_size, 16, features)
print(type(last_16_timesteps))

# 打印结果形状
print(f"最后16个时间步的数据形状：{last_16_timesteps.shape}")