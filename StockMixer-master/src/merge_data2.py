import numpy as np

# (83,735,5)
data1 = np.load('stock_data.npy')
# (735,83,3)
data2 = np.load('news_tensor2.npy')
data2 = np.transpose(data2, (1, 0, 2))
part1 = data1[:, :, :4]
part2 = data2
part3 = data1[:, :, -1:]
# (83, 42, 8)
merged_data = np.concatenate([part1, part2, part3], axis=2)
np.save('merged_data2.npy', merged_data)
print("合并后数据形状:", merged_data.shape)