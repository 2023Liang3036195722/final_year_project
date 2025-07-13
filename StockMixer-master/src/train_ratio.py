import numpy as np
import torch as torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from model import StockMixer,get_loss
from dataset import HKStockDataset
import wandb
from scipy import stats


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.")
else:
    print("CUDA not available!")
    exit()

def compute_metrics(outputs, labels):
    B, N = outputs.shape
    outputs = outputs.cpu().numpy()
    labels = labels.cpu().numpy()
    # 1. 计算MSE（均方误差）
    # 方法：先计算每个元素的平方差，再对所有元素取平均（或按批次平均后再取总平均）
    mse = np.mean((outputs - labels) ** 2)
    # 2. 计算平均Pearson相关系数
    # 方法：对每个批次（B个）计算a和b的Pearson系数，再取平均值
    # 注：scipy.stats.pearsonr返回（相关系数，p值），需提取相关系数
    pearson_coeffs = [stats.pearsonr(outputs[i], labels[i])[0] for i in range(B)]
    IC = np.mean(pearson_coeffs)
    # 3. 计算平均Spearman相关系数
    # 方法：类似Pearson，对每个批次计算Spearman系数后取平均
    # 注：Spearman是基于排序的非参数相关系数
    spearman_coeffs = [stats.spearmanr(outputs[i], labels[i])[0] for i in range(B)]
    RIC = np.mean(spearman_coeffs)
    # 4. 计算Precision@10
    # 方法：对每个批次，找出outputs预测的前10个最大值的索引，计算这些索引在labels的前10个最大值索引中的比例
    k = 10
    precisions = [np.mean(np.isin(np.argsort(outputs[i])[-k:], np.argsort(labels[i])[-k:])) for i in range(B)]
    prec_10 = np.mean(precisions)
    return mse,IC,RIC,prec_10

def train_test(config=None):
    with wandb.init(config=config):
        config = wandb.config
        lr = config.lr
        hyper_m = config.hyper_m
        weight_decay = config.weight_decay

        np.random.seed(123456789)
        torch.random.manual_seed(12345678)
        # # 超参数探索实验设置
        # # 7f5f63654c990eb6c7d796f534bf9db8cfcae73e
        # # 学习率
        # lr = 1e-3
        # hyper_m = 10
        # import wandb
        # wandb.login()
        # run = wandb.init(
        #     project="stock",
        #     name = "stock_200",
        #     config={
        #         "lr": lr,
        #         "hyper_m": hyper_m
        #     },
        # )

        # 基本信息
        stock_num = 83
        windows_length = 16
        fea_num = 5
        market_num = hyper_m
        scale_factor = 3


        data = np.load('stock_data.npy')
        dataset = HKStockDataset(data)

        train_size = int(0.90 * len(dataset))
        print("train_size:", train_size)
        test_size = len(dataset) - train_size
        print("test_size:", test_size)

        train_dataset = Subset(dataset, range(train_size))
        test_dataset = Subset(dataset, range(train_size, len(dataset)))

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        model = StockMixer(
            stocks=stock_num,
            time_steps=windows_length,
            features=fea_num,
            market=market_num,
            scale=scale_factor
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        print('begin training...')
        total_train_batches = len(train_dataloader)
        total_test_batches = len(test_dataloader)
        print('total train batches'+str(total_train_batches))
        print('total test_batches'+str(total_test_batches))


        num_epochs = 100
        best_test_loss = float('inf')

        for epoch in range(num_epochs):
            # train
            model.train()
            running_loss = 0.0
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = get_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            running_loss = running_loss / total_train_batches

            # test
            model.eval()
            test_loss = 0.0
            mse_list, IC_list, RIC_list, prec_10_list = [], [], [], []
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = get_loss(outputs, labels)
                    test_loss += loss.item()
                    mse,IC,RIC,prec_10=compute_metrics(outputs, labels)
                    mse_list.append(mse)
                    IC_list.append(IC)
                    RIC_list.append(RIC)
                    prec_10_list.append(prec_10)
            test_loss = test_loss / total_test_batches
            mse = np.mean(mse_list)
            IC = np.mean(IC_list)
            RIC = np.mean(RIC_list)
            prec_10 = np.mean(prec_10_list)

            # 打印所有指标到控制台
            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'running_loss: {running_loss:.6f}, '
                  f'test_loss: {test_loss:.6f}, '
                  f'MSE: {mse:.6f}, '
                  f'IC: {IC:.6f}, '
                  f'RIC: {RIC:.6f}, '
                  f'Precision@10: {prec_10:.6f}')

            # 将所有指标同步到W&B
            wandb.log({
                "running_loss": running_loss,
                "test_loss": test_loss,
                "MSE": mse,
                "IC": IC,
                "RIC": RIC,
                "Precision@10": prec_10
            })

            # 保存最佳模型
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Best model saved at epoch {epoch + 1} with test_loss: {best_test_loss:.6f}")

if __name__ == '__main__':
    # Hyperparameter Exploration
    wandb.login(key='7f5f63654c990eb6c7d796f534bf9db8cfcae73e')
    # 扫描
    sweep_configuration = \
        {
            'name': 'e1',
            'early_terminate': {'eta': 2, 'min_iter': 30, 's': 3, 'type': 'hyperband'},
            'method': 'bayes',
            'metric': {'goal': 'minimize', 'name': 'test_loss'},
            'parameters':
                {
                    'hyper_m': {'values': [10,15,20] },
                    'lr': {'distribution': 'uniform', 'min': 5e-3, 'max': 5e-2},
                    'weight_decay': {'distribution': 'uniform', 'max': 1e-4, 'min': 1e-5}
                }
        }

    # # 运行
    # sweep_configuration = \
    #     {
    #         'name': 'run_best',
    #         'method': 'bayes',
    #         'metric': {'goal': 'minimize', 'name': 'test_sqrt_loss'},
    #         'parameters':
    #             {
    #                 'hyper_m': {'values': [15] },
    #                 'lr': {'values': [0.004746203357144903]},
    #                 'weight_decay': {'values': [2.6812795189241473e-05]}
    #             }
    #     }

    # Initialize sweep by passing in config.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="stock")
    # Start sweep job.
    wandb.agent(sweep_id, function=train_test, count=5)