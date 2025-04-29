# NS-2025-05 题解 —— Tokisakix 的出行计划

## 1 概述

本题目要求基于 2019 年 10 月到 2024 年 12 月的历史交通流量数据，预测 2025 年 1 月 1 日 00:00 到 2025 年 9 月 30 日 23:00 期间，每一个小时的交通流量。

由于预测时间跨度长达 9 个月，流量数据具有明显的趋势性、周期性和外部特征依赖性，因此不能**简单直接地套用**传统时间序列模型。

本题推荐的整体方法是：

首先使用时序模型预测未来每一天的重要特征（如天气、温度等），然后基于这些特征，再用回归模型预测每小时的交通流量。

这样的分步策略，可以有效避免长周期预测中误差累计的问题。

根据赛后的 WP 情况来看，大部分的队伍的解法被束缚在纯时序模型中，有一两支队伍采用了本题推荐的整体方法。

## 2 题解

### 2.1 交通流量预测

**1. 直接预测数值**

本题最简单直接的方法是，把每一个小时的交通流量当作一个回归目标，直接建立模型进行预测。

训练数据集由每个时刻的各种特征组成（如小时数、星期几、是否节假日、上一个小时流量等），输出为对应时刻的流量。

例如，可以使用随机森林进行直接建模：

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

这种方法实现简单，但由于不考虑流量变化的连续性和累积性，长期预测时容易出现偏差。

**2. 预测涨跌幅度**

另一种策略是预测交通流量的变化率（涨跌幅度），而不是直接预测绝对数值。

定义涨跌幅度：

\[
r_t = \frac{y_t - y_{t-1}}{y_{t-1}}
\]

其中 \( y_t \) 是第 \( t \) 个小时的交通流量。

先预测 \( r_t \)，然后用递推的方法恢复出 \( y_t \)：

```python
r_train = (y_train[1:] - y_train[:-1]) / y_train[:-1]

model = RandomForestRegressor()
model.fit(X_train[:-1], r_train)

r_pred = model.predict(X_test)
y_pred = [y_train[-1]]
for r in r_pred:
y_pred.append(y_pred[-1] * (1 + r))
```

这种方法能更好地捕捉流量变化趋势，但需要注意累积误差的问题。

### 2.2 多特征预测

在建模之前，需要对原始数据进行特征工程。除了数据集自带的天气、气温等特征，还可以从数据集提取以下常见的特征：

- 小时（0-23）
- 星期几（0-6）
- 是否周末
- 月份（1-12）
- 是否节假日
- 前1小时、前24小时的流量（滞后特征）

使用Pandas可以方便地提取这些特征：

```python
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month'] = df['timestamp'].dt.month
df['lag_1'] = df['traffic'].shift(1)
df['lag_24'] = df['traffic'].shift(24)
```

滞后特征需要注意填充或去除NaN值。

考虑到未来特征（如天气、温度）未知，我们需要首先使用时序模型对这些特征进行预测。

可以采用 Prophet、SARIMA、LSTM 等模型来对每天的天气、温度、降水量等特征序列进行建模。

例如使用 Prophet 预测未来温度：

```python
from prophet import Prophet

df_temp = df[['timestamp', 'temperature']].rename(columns={'timestamp': 'ds', 'temperature': 'y'})
model = Prophet()
model.fit(df_temp)
future = model.make_future_dataframe(periods=periods, freq='D')
forecast_temp = model.predict(future)
future_temperature = forecast_temp['yhat']
```

通过这种方式，得到 2025 年每天的特征预测结果，供后续流量预测使用。

值得注意的是：

时序模型适合短期预测，长期预测容易出现误差累积。

特别是在九个月的长时间段内，外部环境的变化（如季节、节假日）会导致单纯时序模型预测结果偏离实际。

**因此我们不直接用时序模型预测交通流量，而是只用它预测每天的环境特征。**

### 2.3 回归模型

交通流量并不是一个真正意义上的时序数据，它与当天的特征（如天气、日期、温度）密切相关。

当天的特征之间构成了时序关系。

今天的温度会受前一天影响

今天是否节假日是确定的

今天的降雨量可以由时序模型提前预测

因此，合理的建模策略是：

**用时序模型先预测 2025 年每天的特征，再用回归模型根据这些特征预测交通流量。**

建模公式可以表示为：

1. 特征预测（时序模型）：

\[
\hat{x}_t = f(x_{t-1}, x_{t-2}, \ldots)
\]

2. 流量预测（回归模型）：

\[
\hat{y}_{t,h} = g(\hat{x}_t, h, \text{lag features}, \text{date features})
\]


LightGBM（Light Gradient Boosting Machine）是一种基于梯度提升决策树（GBDT）的高效框架，专门为大规模数据和高效训练设计。它的基本原理是：

1. 每一棵新树学习当前模型残差（即预测误差）。
2. 新树的目标是减少整体损失函数，例如均方误差（MSE）：

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

3. LightGBM 很适合处理离散特征。

在本题中，未来九个月每天的特征已经由时序模型提前预测出来，因此我们要解决的问题是：

- 给定未来一天的特征（如温度、降雨、是否节假日、小时数等）
- 预测每小时的交通流量

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000
)

model.fit(X_train, y_train,
          eval_set=[(X_valid, y_valid)],
          early_stopping_rounds=100,
          verbose=100)

y_pred = model.predict(X_test)
```

其中 `X_train` 包含选手所有提取出来的特征，`y_train` 是对应时刻的交通流量。

### 2.4 周期因子

**数据预处理**在本题很重要，如何更好地表示或引入更多的特征是解题的关键。

不难发现，交通流量具有明显的周期性变化：

- 每天的早高峰（如 7-9 点）、晚高峰（如 17-19 点）
- 每周的工作日（周一到周五）、周末（周六周日）流量差异
- 每月的月底流量波动
- 每年的季节性变化

因此，在特征工程阶段，我们可以刻意额外添加周期性特征，帮助模型更好地理解这些规律。

**1. 使用正余弦变换捕捉周期性**

时间类特征（如小时、星期几、月份）本质上是一个**环状（循环的）变量**，比如 23 点和 0 点实际上很近，而不是差了 23 小时。

直接使用数值编码（如 hour=23，hour=0）会误导模型，因此我们使用**正余弦变换**：

```python
import numpy as np

# 小时的周期特征
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# 星期几的周期特征
df['dayofweek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# 月份的周期特征（如果需要）
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

正余弦特征可以帮助模型捕捉“接近性”，例如 23 点和 0 点、周日和周一的连续性。

**2. 在模型中直接引入周期性描述子**

除了手动构造特征，此题更推荐直接使用带有内置周期性建模能力的工具，比如 Prophet 模型。

Prophet 模型将时间序列分解为：

\[
y(t) = g(t) + s(t) + h(t) + \epsilon_t
\]

Prophet 的一个优点是，只需要提供时间戳和流量值，它会自动拟合出周期规律，无需手动特征工程。

```python
from prophet import Prophet

df_prophet = df[['ds', 'y']]
model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=24*30)
forecast = model.predict(future)
```