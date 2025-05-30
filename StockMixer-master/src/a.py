import numpy as np

# 读取 .npy 文件
data = np.load('news_tensor.npy')

# 查看数据的基本信息
print('数据类型：', type(data))
print('数据形状：', data.shape)
print('数据内容：')
print(data)