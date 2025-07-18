import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the .npy file
# data = np.load('stock_data.npy')
data = np.load('stock_data_normalized.npy')
print('Data shape:', data.shape)
print('Data content:')
print(data)
features = ['High', 'Low', 'Open', 'Volume', 'Close']

# # 1. Analyze the distribution by feature
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# axes = axes.flatten()
#
# for i in range(data.shape[2]):
#     feature_data = data[:, :, i].flatten()
#     # Calculate statistics
#     mean_val = np.mean(feature_data)
#     std_val = np.std(feature_data)
#     min_val = np.min(feature_data)
#     max_val = np.max(feature_data)
#     # Plot distribution with better spacing
#     sns.histplot(feature_data, kde=True, bins=50, ax=axes[i])
#     # Set title with statistical information
#     axes[i].set_title(f'{features[i]} \n(Mean={mean_val:.4f}, STD={std_val:.4f})', fontsize=12)
#     axes[i].set_xlabel('Value', fontsize=10)
#     axes[i].set_ylabel('Frequency', fontsize=10)
#     # Add annotation with min/max values
#     axes[i].annotate(f'Min: {min_val:.4f}\nMax: {max_val:.4f}',
#                      xy=(0.5, 0.95), xycoords='axes fraction',
#                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
#                      fontsize=9, ha='left', va='top')
# # Remove empty subplots
# fig.delaxes(axes[5])
# # Adjust layout spacing
# plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.35)
# plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
# plt.show()

# 2. Analyze the distribution by feature ----single stock 12
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for i in range(data.shape[2]):
    feature_data = data[12, :, i].flatten()
    # Calculate statistics
    mean_val = np.mean(feature_data)
    std_val = np.std(feature_data)
    min_val = np.min(feature_data)
    max_val = np.max(feature_data)
    # Plot distribution with better spacing
    sns.histplot(feature_data, kde=True, bins=50, ax=axes[i])
    # Set title with statistical information
    axes[i].set_title(f'{features[i]} \n(Mean={mean_val:.4f}, STD={std_val:.4f})', fontsize=12)
    axes[i].set_xlabel('Value', fontsize=10)
    axes[i].set_ylabel('Frequency', fontsize=10)
    # Add annotation with min/max values
    axes[i].annotate(f'Min: {min_val:.4f}\nMax: {max_val:.4f}',
                     xy=(0.5, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     fontsize=9, ha='left', va='top')
# Remove empty subplots
fig.delaxes(axes[5])
# Adjust layout spacing
plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.35)
plt.savefig('feature_distributions_stock12.png', dpi=600, bbox_inches='tight')
plt.show()

# 3. Time trend analysis
plt.figure(figsize=(15, 10))
for i in range(data.shape[2]):
    plt.subplot(3, 2, i + 1)
    plt.plot(data[12, :, i])
    plt.title(f'{features[i]} Time trend (stock12 0386.HK)')
    plt.xlabel('Time point')
    plt.ylabel('Value')
plt.tight_layout()
plt.savefig('time_trends_stock12.png')
plt.show()

# 4. Outlier detection (using the IQR method)
plt.figure(figsize=(15, 8))
for i in range(data.shape[2]):
    feature_data = data[12, :, i].flatten()
    # Calculate quartiles
    q1 = np.percentile(feature_data, 25)
    q3 = np.percentile(feature_data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Detect outliers
    outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
    print(f"\n{features[i]} Number of outliers: {len(outliers)} ({len(outliers) / len(feature_data) * 100:.2f}%)")
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=feature_data)
    plt.title(f'{features[i]} Box plot (stock12 0386.HK)')
plt.tight_layout()
plt.savefig('boxplots_stock12.png')
plt.show()

# 5. Feature correlation analysis
plt.figure(figsize=(10, 8))
# Calculate the feature correlation for all samples at all time points
correlation = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        if i != j:
            # Calculate the Pearson correlation coefficient between two features
            corr = np.corrcoef(data[:, :, i].flatten(), data[:, :, j].flatten())[0, 1]
            correlation[i, j] = corr
# Draw a heatmap
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
            xticklabels=features, yticklabels=features)
plt.title('Feature correlation analysis (stock12 0386.HK)')
plt.savefig('feature_correlation_stock12.png')
plt.show()