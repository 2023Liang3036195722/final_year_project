import numpy as np
import torch as torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from model import StockMixer,get_loss
from dataset import HKStockDataset
from scipy import stats


def compute_metrics(outputs, labels):
    B, N = outputs.shape
    performance = {}
    # 1. 计算MSE（均方误差）
    # 方法：先计算每个元素的平方差，再对所有元素取平均（或按批次平均后再取总平均）
    performance['mse'] = np.mean((outputs - labels) ** 2)
    # 2. 计算平均Pearson相关系数
    # 方法：对每个批次（B个）计算a和b的Pearson系数，再取平均值
    # 注：scipy.stats.pearsonr返回（相关系数，p值），需提取相关系数
    pearson_coeffs = np.array([stats.pearsonr(outputs[i], labels[i])[0] for i in range(B)])
    performance['IC'] = np.mean(pearson_coeffs)
    # 3. 计算平均Spearman相关系数
    # 方法：类似Pearson，对每个批次计算Spearman系数后取平均
    # 注：Spearman是基于排序的非参数相关系数
    spearman_coeffs = np.array([stats.spearmanr(outputs[i], labels[i])[0] for i in range(B)])
    performance['RIC'] = np.mean(spearman_coeffs)
    # 4. 计算Precision@10
    # 方法：对每个批次，找出outputs预测的前10个最大值的索引，计算这些索引在labels的前10个最大值索引中的比例
    k = 10
    precisions = [np.mean(np.isin(np.argsort(outputs[i])[-k:], np.argsort(labels[i])[-k:])) for i in range(B)]
    performance['prec_10'] = np.mean(precisions)
    return performance

if __name__ == '__main__':
    data = np.load('stock_data.npy')
    timesteps = data[:, 10:26, :]
    print(timesteps.shape)
    x = data[:, 25, -1]
    y = data[:, 26, -1]
    predict_result = predict(timesteps)
    true_result = (y-x)/x
    print('predict:',predict_result)
    print('true:',true_result)