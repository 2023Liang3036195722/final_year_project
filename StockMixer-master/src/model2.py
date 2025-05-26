import torch
import torch.nn as nn
import torch.nn.functional as F

acv = nn.Hardswish()

# 输入为(B,N,T,F)
# 输出为若干(B,N,T/k,F)  k=1,2,4...
def segment_time(x, scale):
    segments = [x]
    x = x.permute(0, 1, 3, 2).contiguous()
    # B,N,F,T
    for i in range(1, scale + 1):
        k = 2 ** i
        Avgpool_x = F.avg_pool1d(x.view(x.shape[0], -1, x.shape[3]), kernel_size=k, stride=k)
        # B,N*F,T
        Avgpool_x = (Avgpool_x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3] // k).
                     permute(0, 1, 3, 2).contiguous())
        # B,N,T/k,F
        segments.append(Avgpool_x)
    return segments

class IndicatorMixer(nn.Module):
    def __init__(self, time_steps, features):
        super(IndicatorMixer, self).__init__()
        self.features = features
        hidden_dim = features
        self.LN = nn.LayerNorm([time_steps, features])
        self.dense1 = nn.Linear(features, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, features)

    def forward(self, inputs):
        inputs = self.LN(inputs)
        x = inputs.view(-1, self.features)
        x = self.dense1(x)
        x = acv(x)
        x = self.dense2(x)
        x = x.view(inputs.shape)
        y = inputs+x
        return y

# 已经改写
# (B*N*F,T)
class TriU(nn.Module):
    def __init__(self, time_step):
        super(TriU, self).__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList([nn.Linear(i, 1) for i in range(1,time_step+1)])

    def forward(self, inputs):
        x = self.triU[0](inputs[:, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, 0:i+1])], dim=-1)
        return x

class TimeMixer(nn.Module):
    def __init__(self, time_steps, features):
        super(TimeMixer, self).__init__()
        self.time_steps = time_steps
        self.LN = nn.LayerNorm([time_steps, features])
        self.u1 = TriU(time_steps)
        self.u2 = TriU(time_steps)

    def forward(self, inputs):
        inputs = self.LN(inputs)
        inputs_T = inputs.permute(0, 1, 3, 2).contiguous()
        # inputs_T (B,N,F,T)
        x = inputs_T.view(-1, self.time_steps)
        # x (B*N*F,T)
        x = self.u1(x)
        x = acv(x)
        x = self.u2(x)
        x = x.view(inputs_T.shape)
        y = inputs_T+x
        y = y.permute(0, 1, 3, 2).contiguous()
        return y

# markets是市场超参数
class MarketMixer(nn.Module):
    def __init__(self, stocks, markets):
        super(MarketMixer, self).__init__()
        self.stocks = stocks
        hidden_dim = markets
        self.LN = nn.LayerNorm(stocks)
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, stocks)

    def forward(self, inputs):
        inputs_T = inputs.permute(0, 2, 1).contiguous()
        # inputs_T (B,d,N)
        inputs_T = self.LN(inputs_T)
        x = inputs_T.view(-1, self.stocks)
        x = self.dense1(x)
        x = acv(x)
        x = self.dense2(x)
        x = x.view(inputs_T.shape)
        y = inputs_T+x
        y = y.permute(0, 2, 1).contiguous()
        return y


class StockMixer(nn.Module):
    def __init__(self, stocks, time_steps, features, market, scale):
        super(StockMixer, self).__init__()
        self.scale = scale
        self.indicator_mixer = nn.ParameterList([IndicatorMixer(time_steps//(2**i), features) for i in range(scale)])
        self.time_mixer = nn.ParameterList([TimeMixer(time_steps//(2**i), features) for i in range(scale)])
        d = sum([time_steps//(2**i) for i in range(scale)])
        self.fc1 = nn.Linear(features, 1)
        self.market_mixer = MarketMixer(stocks, market)
        self.fc2 = nn.Linear(d, 1)


    def forward(self, inputs):
        # print(inputs.shape)
        # print('______________')
        x_k_list = segment_time(inputs,self.scale)
        h_k_list = []
        for i in range(self.scale):
            x_k = self.indicator_mixer[i](x_k_list[i])
            h_k = self.time_mixer[i](x_k)
            h_k_list.append(h_k)
        x = torch.cat(h_k_list, dim=2)
        # x (B,N,d,F) d=T+T/2+T/4+...
        x = self.fc1(x).squeeze(-1)
        # x(B,N,d,1)  x(B,N,d)
        x = self.market_mixer(x)
        # x(B,N,d)
        x = self.fc2(x).squeeze(-1)
        # x(B,N)
        # print(x.shape)
        return x
