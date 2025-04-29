# NS-2025-03 “净语行动·进阶计划”题解 —— CLEAN II

## 问题描述
本项目旨在构建第二代「净语系统」，对中文评论同时完成以下任务：
- **细粒度冒犯性检测**（4 类：0 安全、1 个人攻击、2 群体攻击、3 反偏见）；
- **敏感话题识别**（多标签：race, gender, region）。

训练集仅包含粗粒度标签（0/1）和话题标签，需通过大模型伪标、局部微调与多任务学习实现目标。

---

## Pipeline 概述
1. **数据清洗**：去噪、简繁一体、统一长度；  
2. **伪标签生成**：利用本地 Qwen2.5-7B-instruct zero/few-shot 批量打细粒度与话题标签；  
3. **特征构建**：汉字级 BERT Tokenizer 处理文本，并可选拼接 Qwen 隐层均池向量；  
4. **多任务模型**：基于中文 BERT 添加两个头——一个 4 类 Softmax、一个 3 维 Sigmoid；  
5. **训练策略**：先二分类预热，再多任务微调，使用交叉熵与 BCE 混合损失，加入 FGM 对抗；  
6. **推理与融合**：本地模型与 Qwen 输出加权融合，阈值决策，生成提交文件。

---

## 设计思路
- **伪标签**：充分利用 Qwen 的 zero/few-shot 强大分类能力，解决细粒度样本匮乏问题；  
- **多任务并行**：共享编码器减少参数、利用话题辅助信号提升泛化；  
- **本地化部署**：全程依赖开源 Qwen 与 BERT，无需外部闭源 API，易于落地与迭代；  
- **对抗与正则化**：FGM、随机掩码等增强模型对隐晦措辞的鲁棒性；  
- **加权融合**：结合 Qwen 与本地模型决策，兼顾大模型语言理解和轻量微调的高效性。

---

## 核心代码示例

```python
# 伪标签生成（示例）
def pseudo_label(text, prompt):
    ans = qwen_model.generate(qwen_tok(prompt.format(text), return_tensors='pt'))
    return int(ans.decode().strip()[0])

# 多任务模型定义
class MultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm')
        self.fc_off = nn.Linear(768, 4)
        self.fc_top = nn.Linear(768, 3)

    def forward(self, x, attn, y_off=None, y_top=None):
        h = self.bert(x, attn).pooler_output
        loff = self.fc_off(h)
        ltop = self.fc_top(h)
        if y_off is not None:
            loss = 0.7 * F.cross_entropy(loff, y_off) \
                 + 0.3 * F.binary_cross_entropy_with_logits(ltop, y_top)
            return loss
        return loff, ltop

# 训练调用略（Trainer + FGM 对抗 + EarlyStop）
```

## 总结

本方案结合 Qwen2.5-7B-instruct 的强零样本能力与中文 BERT 的高效微调，通过多任务框架与对抗正则化，解决了细粒度标签稀缺和双任务联合的问题，实现了可部署、可迭代、效果稳健的第二代「净语系统」。