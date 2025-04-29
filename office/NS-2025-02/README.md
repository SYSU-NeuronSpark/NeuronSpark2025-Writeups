# NS-2025-02 “净语行动”题解 —— CLEAN I

## 问题描述
本项目旨在构建一个二分类模型，对中文游戏评论进行审核，识别其中的负面（违规）评论。数据集中正常评论占约85%，违规（侮辱、歧视、攻击性）评论占约15%，存在明显的类别不平衡，且评论中充斥大量游戏术语、俚语及隐晦表达。

## 模型架构
本方案选择基于预训练的中文BERT模型 `hfl/chinese-bert-wwm`，在其上添加一个简单的分类头（classification head）输出二分类结果。分类头由一个全连接层（linear）构成，将 BERT 最后一层 [CLS] token 的向量映射到 2 维预测空间。

## LoRA微调与分类头的选择理由
- **参数高效微调：** LoRA（Low-Rank Adaptation）仅在部分权重矩阵上学习低秩更新矩阵，大幅减少了需训练的参数量，从而降低显存占用与训练成本。
- **适应小样本与类别不平衡：** 在只有约15%违规样本的情况下，LoRA 能更有效地利用有限数据学习语义差异，而不至于因全参数微调过拟合常见样本。
- **保持预训练知识：** LoRA 不修改原模型参数，只叠加低秩修正，能够最大程度保留 BERT 在大规模语料上学到的语言理解能力，特别是对隐晦俚语的捕捉。
- **简单高效的分类头：** 在 BERT 上叠加单层全连接分类头，能快速收敛且具有良好的泛化能力，且在工程部署时易于实现实时推理。

## 核心代码实现
```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# 1. 加载数据
train_ds = load_dataset('csv', data_files='train_set.csv', split='train')
test_ds  = load_dataset('csv', data_files='test_set.csv',  split='test')

# 2. 分词器与模型
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')
model = BertForSequenceClassification.from_pretrained(
    'hfl/chinese-bert-wwm', num_labels=2
)

# 3. 数据预处理
def tokenize_fn(ex):
    return tokenizer(ex['text'], padding='max_length', truncation=True)
train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds  = test_ds.map(tokenize_fn,  batched=True)

# 4. 配置 LoRA
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias='none'
)
model = get_peft_model(model, lora_cfg)

# 5. 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# 6. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer
)
trainer.train()

# 7. 保存微调后模型
model.save_pretrained('./final_model')
```

## 模型评估与预测
训练完成后，可使用以下代码进行预测并保存结果：

```python
# 预测
preds = trainer.predict(test_ds)
labels = preds.predictions.argmax(axis=-1)

# 输出至 JSON
import json
results = [
    {"text": txt, "label": int(lbl)}
    for txt, lbl in zip(test_ds['text'], labels)
]
with open('predictions.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
```

## 总结

本方案通过在中文BERT基础上加入 LoRA 微调与轻量分类头，实现了对游戏评论中负面内容的高效识别。在类别不平衡和游戏专属表达环境下，LoRA 的低秩适配既保障了模型性能，又降低了训练与部署成本，满足实时检测的需求。