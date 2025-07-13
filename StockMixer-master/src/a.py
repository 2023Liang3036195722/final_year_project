import numpy as np
# 读取 .npy 文件
data = np.load('SP500.npy')

# 查看数据的基本信息
print('数据类型：', type(data))
print('数据形状：', data.shape)
print('数据内容：')
print(data)

# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 设置中文字体
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
#
# # 加载数据
# data = np.load('SP500.npy')
# print(f"数据形状: {data.shape} (样本数, 时间点, 特征数)")
#
# # 1. 整体统计信息
# print("\n=== 整体统计信息 ===")
# print(f"数据类型: {data.dtype}")
# print(f"整体最小值: {np.min(data):.4f}")
# print(f"整体最大值: {np.max(data):.4f}")
# print(f"整体均值: {np.mean(data):.4f}")
# print(f"整体标准差: {np.std(data):.4f}")
#
# # 2. 按特征分析分布
# features = ['特征1', '特征2', '特征3', '特征4', '特征5']
# plt.figure(figsize=(15, 10))
#
# for i in range(data.shape[2]):
#     feature_data = data[99, :, i].flatten()
#
#     plt.subplot(2, 3, i + 1)
#     sns.histplot(feature_data, kde=True, bins=50)
#     plt.title(f'{features[i]} 分布')
#     plt.xlabel('值')
#     plt.ylabel('频次')
#
#     # 显示统计信息
#     plt.figtext(0.15, 0.02 + i * 0.04,
#                 f"{features[i]}: 均值={np.mean(feature_data):.4f}, 标准差={np.std(feature_data):.4f}, "
#                 f"最小值={np.min(feature_data):.4f}, 最大值={np.max(feature_data):.4f}",
#                 fontsize=10)
#
# plt.tight_layout()
# plt.savefig('feature_distributions.png')
# plt.show()
#
# # 3. 时间趋势分析（每个特征选择一个代表性样本）
# plt.figure(figsize=(15, 10))
# sample_idx = 0  # 选择第一个样本进行分析
#
# for i in range(data.shape[2]):
#     plt.subplot(3, 2, i + 1)
#     plt.plot(data[sample_idx, :, i])
#     plt.title(f'{features[i]} 时间趋势 (样本 {sample_idx})')
#     plt.xlabel('时间点')
#     plt.ylabel('值')
#
# plt.tight_layout()
# plt.savefig('time_trends.png')
# plt.show()
#
# # 4. 异常值检测（使用IQR方法）
# plt.figure(figsize=(15, 8))
#
# for i in range(data.shape[2]):
#     feature_data = data[:, :, i].flatten()
#
#     # 计算四分位数
#     q1 = np.percentile(feature_data, 25)
#     q3 = np.percentile(feature_data, 75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#
#     # 检测异常值
#     outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
#     print(f"\n{features[i]} 异常值数量: {len(outliers)} ({len(outliers) / len(feature_data) * 100:.2f}%)")
#
#     plt.subplot(2, 3, i + 1)
#     sns.boxplot(y=feature_data)
#     plt.title(f'{features[i]} 箱线图')
#
# plt.tight_layout()
# plt.savefig('boxplots.png')
# plt.show()
#
# # 5. 特征相关性分析
# plt.figure(figsize=(10, 8))
#
# # 计算所有样本所有时间点的特征相关性
# correlation = np.zeros((5, 5))
# for i in range(5):
#     for j in range(5):
#         if i != j:
#             # 计算两个特征之间的Pearson相关系数
#             corr = np.corrcoef(data[:, :, i].flatten(), data[:, :, j].flatten())[0, 1]
#             correlation[i, j] = corr
#
# # 绘制热图
# mask = np.triu(np.ones_like(correlation, dtype=bool))
# sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
#             xticklabels=features, yticklabels=features)
# plt.title('特征相关性分析')
# plt.savefig('feature_correlation.png')
# plt.show()
#
# print("\n=== 分布分析总结 ===")
# print("1. 各特征的基本统计信息（均值、标准差等）已显示在分布图下方")
# print("2. 时间趋势图展示了各特征随时间的变化情况")
# print("3. 异常值检测显示了可能需要处理的离群数据点")
# print("4. 特征相关性热图展示了各特征间的线性相关程度")