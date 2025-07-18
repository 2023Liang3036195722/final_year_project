import numpy as np

 # (83,735,5)
data = np.load('stock_data.npy')
normalized_data = np.zeros_like(data, dtype=np.float32)
max_values = np.zeros((data.shape[0], data.shape[2]))

for i in range(data.shape[0]):
    for k in range(data.shape[2]):
        slice_data = data[i, :, k]
        # volume : log transformation
        if k == 3:
             slice_data = np.log(slice_data + 1)
        max_val = np.max(slice_data)
        normalized_data[i, :, k] = slice_data / max_val
        max_values[i, k] = max_val
np.save('stock_data_normalized.npy', normalized_data)
np.save('stock_data_max_values.npy', max_values)