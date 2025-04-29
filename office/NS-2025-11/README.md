# NS-2025-11 题解 —— 接管厨房客厅的那一天

随着机器人应用领域的不断拓展，单一模态（如仅凭视觉）已无法满足复杂任务的需求。机器人需要同时理解环境（视觉与点云）和高层任务指令（语言），才能完成如堆叠积木、安装灯泡、放置物品等实际操作。本题正是围绕**多模态感知与融合**展开的一个综合训练。

本题解是一个此题**通解的框架**，但仍有不少可以优化的地方。希望同学们在理解之上，积极探索改进的可能。

相关代码可以在同级目录下的 `model.py` 找到。

## 1 模态处理

多模态感知的第一步，就是对每种数据进行独立编码处理。

数据输入格式：
- `obs.npy`：形状 \((N, 4, 2, 3, H, W)\)，表示4视角下RGB和点云。
- `instr.npy`：形状 \((N, 53, 512)\)，语言指令的token序列。
- `action.npy`：形状 \((N, 1, 8)\)，动作数据（位置、旋转、夹爪状态）。

### 1.1 RGB 处理

```python
base_model = models.resnet18(pretrained=True)
self.rgb_encoder = nn.Sequential(*list(base_model.children())[:-1])
```

这里我们使用预训练的**ResNet18**，去掉了分类头，只保留卷积和池化层。每张RGB图像通过网络后，提取出一个**512维的全局特征向量**。

当然此处也可以更换为一个更复杂的图像特征提取网络，比如ViT及其衍生模型。

\[
f_{\text{rgb}} = \text{ResNet18}(x_{\text{rgb}})
\]

每个视角独立处理。

在`model.py`的`VisionEncoder.forward()`中，对每一帧RGB图像调用`self.rgb_encoder(rgb)`进行特征提取，并用`.flatten(1)`展平成(batch, 512)维。

### 1.2 点云处理

点云数据在输入时已经被投影成了类似RGB的格式（通常为伪彩色深度图或点云图）。在代码中，点云与RGB使用**相同的ResNet18结构**，但各自有独立的权重。

```python
self.pcd_encoder = models.resnet18(pretrained=True)
self.pcd_encoder = nn.Sequential(*list(self.pcd_encoder.children())[:-1])
```

\[
f_{\text{pcd}} = \text{ResNet18\_pcd}(x_{\text{pcd}})
\]

在`model.py`的`VisionEncoder.forward()`里，点云也是逐帧提取特征。

### 1.3 文本处理

语言模态的处理使用了**Transformer Encoder**，捕捉token之间的上下文关系。

```python
self.transformer = nn.TransformerEncoder(
nn.TransformerEncoderLayer(d_model=input_dim, nhead=8),
num_layers=2
)
```

- 输入维度是512（与CLIP输出对应）。
- 有两层Transformer编码器，每层有8个头。

Transformer输出后，对token序列取均值（`.mean(dim=1)`），然后映射到256维：

```python
self.proj = nn.Linear(input_dim, hidden_dim)
```

最终得到每条指令的一个256维向量：

\[
f_{\text{text}} = \text{Proj}\left( \frac{1}{T} \sum_{t=1}^T \text{Transformer}(x_t) \right)
\]

这部分在`model.py`的`LanguageEncoder.forward()`中实现。

## 2 模态融合

不同模态的特征提取后，还需要有效地融合起来，为决策模型提供统一的信息表示。

### 2.1 模态对齐

实际上模态融合在多模态领域是一个很重要的子课题，这是只是给出了较为简易的样例实现。

在视觉内部，首先将每帧RGB特征和点云特征拼接在一起：

```python
fused = torch.cat([rgb_feat, pcd_feat], dim=1)# (batch_size, 1024)
```

然后通过一个小型MLP（线性层+ReLU+Dropout）降到512维：

```python
self.fusion = nn.Sequential(
nn.Linear(512*2, 512),
nn.ReLU(),
nn.Dropout(0.3)
)
```

\[
f_{\text{vis}} = \text{Fusion}([f_{\text{rgb}}, f_{\text{pcd}}])
\]

四个视角处理后，取均值聚合：

\[
f_{\text{vis-agg}} = \frac{1}{4} \sum_{i=1}^4 f_{\text{vis},i}
\]

### 2.2 动作空间

视觉特征 \(f_{\text{vis-agg}}\) 和语言特征 \(f_{\text{text}}\) 拼接后：

```python
fused = torch.cat([vis_feat, lang_feat], dim=1)
```

通过融合网络压缩到256维特征：

```python
self.fusion = nn.Sequential(
nn.Linear(512+256, 512),
nn.ReLU(),
nn.Dropout(0.3),
nn.Linear(512, 256),
nn.ReLU()
)
```

接下来，从融合特征预测机器人动作：

- 位置（3维）由`self.position_head`预测。
- 姿态（4维四元数）由`self.rotation_head`预测。
- 夹爪开合（1维0-1值）由`self.gripper_head`预测。

在`model.py`的`MultiModalPolicy.forward()`中：

```python
position = self.position_head(fused)
rotation = self.rotation_head(fused)
gripper = self.gripper_head(fused)
return torch.cat([position, rotation, gripper], dim=1)
```

最终输出一个8维动作向量。

## 3 损失函数

模型输出后，需要与真实动作进行对比，指导模型学习正确行为。

损失函数定义在`model.py`的`hybrid_loss`中。

### 3.1 位置损失

预测位置 \(\hat{p}\) 和真实位置 \(p\) 使用均方误差（MSE）计算：

```python
pos_loss = torch.nn.functional.mse_loss(pred[:, :3], target[:, :3])
```

\[
\mathcal{L}_{\text{pos}} = \| \hat{p} - p \|_2^2
\]

权重是0.7，占总损失的大部分。

### 3.2 位姿损失

预测四元数 \(\hat{q}\) 和真实四元数 \(q\) 也是用均方误差：

```python
rot_loss = torch.nn.functional.mse_loss(pred[:, 3:7], target[:, 3:7])
```

\[
\mathcal{L}_{\text{rot}} = \| \hat{q} - q \|_2^2
\]

权重为0.2。

### 3.3 状态损失

夹爪状态预测为一个在0-1之间的概率，使用二分类交叉熵（BCE）计算：

```python
gripper_loss = torch.nn.functional.binary_cross_entropy(pred[:, 7], target[:, 7])
```

\[
\mathcal{L}_{\text{grip}} = -g \log(\hat{g}) - (1-g) \log(1-\hat{g})
\]

权重是0.1。

### 总体损失

总损失是三项加权求和：

```python
return 0.7 * pos_loss + 0.2 * rot_loss + 0.1 * gripper_loss
```

\[
\mathcal{L}_{\text{total}} = 0.7 \mathcal{L}_{\text{pos}} + 0.2 \mathcal{L}_{\text{rot}} + 0.1 \mathcal{L}_{\text{grip}}
\]

## 4 提分技巧

在掌握了基础方法之后，如果希望在任务表现上进一步拔高分数，可以尝试以下**实用且高效**的策略：

**4.1 每个任务单独训练一个模型**

虽然本题目提供了统一的数据接口，但仔细观察可以发现：

不同任务（如 "put_money_in_safe"、"stack_blocks" 等）的视觉场景、动作模式、甚至语言指令分布**差异很大**。

如果直接用一个模型兼顾所有任务，模型必须同时处理多种非常不同的动作逻辑，学习压力极大，容易出现**任务间干扰**，导致性能下降。

因此，一个很有效的提分方法是：

**为每个任务单独训练一套模型参数。**

具体做法：

- 针对每个任务目录，单独构建数据集。
- 初始化独立的模型，分别训练。
- 测试时，按任务调用对应的专属模型。

这样可以让每个模型专注学习本任务的最优策略，从而取得更高准确率和更自然的动作轨迹。

**4.2 先统一训练编码器，再每个任务单独微调训练**

虽然任务差异较大，但视觉感知（图像、点云）和语言理解的底层特征是**有一定共性的**。

- 都需要识别物体的位置、形状。
- 都要理解基本的动作指令（如"放入"、"取出"、"安装"等）。

因此，可以采用如下**两阶段训练策略**，进一步提升效果：

首先利用所有任务的数据，训练视觉编码器（VisionEncoder）和语言编码器（LanguageEncoder）。

然后冻结（或小幅微调）编码器参数。

最后再每个任务上分别训练动作预测模块（MultiModalPolicy的后半部分）。