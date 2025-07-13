import random
import numpy as np
import torch as torch
import wandb

from evaluator import evaluate
from model import get_loss, StockMixer

np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

stock_num = 83
lookback_length = 16
epochs = 30
valid_index = 515 # 时间序列*0.7
test_index = 588 # 时间序列*0.8
features_num = 5
steps = 1
scale_factor = 3

# market_num = 10
# learning_rate = 1e-3
# alpha = 0.1


# (83,735,5)
data = np.load('stock_data_normalized.npy')
close_price_data = data[:, :, -1]
# 没有缺失数据，所有数据有效
mask_data = np.ones((data.shape[0], data.shape[1]))

# 计算一日收益率
one_day_return_data = np.zeros((data.shape[0], data.shape[1]))
for ticket in range(0, data.shape[0]):
    for row in range(1, data.shape[1]):
        one_day_return_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                               data[ticket][row - steps][-1]

trade_dates = mask_data.shape[1]
model = StockMixer(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=features_num,
    market=market_num,
    scale=scale_factor
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_valid_loss = np.inf
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)


def validate(start_index, end_index):
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     stock_num, alpha)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return loss, reg_loss, rank_loss, cur_valid_perf

def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(close_price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(one_day_return_data[:, offset + seq_len + steps - 1], axis=1))

def main():
    for epoch in range(epochs):
        print("epoch{}------".format(epoch + 1))
        np.random.shuffle(batch_offsets)
        tra_loss = 0.0
        tra_reg_loss = 0.0
        tra_rank_loss = 0.0
        for j in range(valid_index - lookback_length - steps + 1):
            data_batch, mask_batch, close_price_batch, one_day_return_batch = map(lambda x: torch.Tensor(x).to(device),
                                                                get_batch(batch_offsets[j]))
            optimizer.zero_grad()
            predict_prices = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(predict_prices, one_day_return_batch,
                                                                close_price_batch, mask_batch, stock_num, alpha)
            cur_loss.backward()
            optimizer.step()
            tra_loss += cur_loss.item()
            tra_reg_loss += cur_reg_loss.item()
            tra_rank_loss += cur_rank_loss.item()
        tra_loss = tra_loss / (valid_index - lookback_length - steps + 1)
        tra_reg_loss = tra_reg_loss / (valid_index - lookback_length - steps + 1)
        tra_rank_loss = tra_rank_loss / (valid_index - lookback_length - steps + 1)
        train_log = 'Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss)
        print(train_log)

        val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
        valid_log = 'Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss)
        print(valid_log)

        test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
        test_log = 'Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss)
        print(test_log)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            # 保存模型
            model_save_path = 'best_model.pth'
            torch.save(model.state_dict(), model_save_path)
            save_log = f"Best model saved to {model_save_path}"
            print(save_log)

        valid_perf_log = ('Valid performance: mse:{:.2e}, IC:{:.2e}, ICIR:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.
            format(val_perf['mse'], val_perf['IC'], val_perf['ICIR'], val_perf['prec_10'], val_perf['sharpe5']))
        print(valid_perf_log)

        test_perf_log = ('Test performance: mse:{:.2e}, IC:{:.2e}, ICIR:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.
            format(test_perf['mse'], test_perf['IC'], test_perf['ICIR'], test_perf['prec_10'], test_perf['sharpe5']))
        print(test_perf_log)



if __name__ == '__main__':
    # Hyperparameter Exploration
    wandb.login(key='7f5f63654c990eb6c7d796f534bf9db8cfcae73e')
    # 扫描
    sweep_configuration = \
        {
            'name': 'e1',
            'early_terminate': {'eta': 2, 'min_iter': 30, 's': 3, 'type': 'hyperband'},
            'method': 'bayes',
            'metric': {'goal': 'maximize', 'name': 'best_prec_10'},
            'parameters':
                {
                    'hyper_m': {'values': [10,15,20] },
                    'lr': {'distribution': 'uniform', 'min': 5e-3, 'max': 5e-2},
                    'alpha' :{'distribution': 'uniform', 'min': 1e-3, 'max': 1e-2},
                }
        }
    # Initialize sweep by passing in config.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="stock")
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=5)