# NS-2025-07 题解 —— LLM to BTs

题解作者：still_dreaming

出题人、验题人、文案设计等：中山大学互联网与开源技术协会（MSC）

## 理解题目

    本题的题干内容，是比较值得细细阅读、取舍和思考的。如果先慢下来将 自己要做什么，要用什么方法来完成 给考虑清楚，完成这一个题目的过程将会更加一日千里。

    本题虽然属于“challenging”难度，但其实题目逻辑并不复杂。整体逻辑已经明确指出：部署好你准备使用的LLM——使用`training_inputs.json` 与 `training_outputs.json` 训练或微调你的LLM——让LLM对 `inputs.json`进行处理得到最终的 `results.xml`.

    相对于本次赛事的其它题目，本题当中涉及到较多可能相对陌生的名词，例如行为树的相关概念、xml格式文件的相关概念、BehaviorTree.CPP库、Groot可视化工具等，它们背后包含着的知识是相对复杂和广大的。或许你初见时对他们并不太熟悉，点开链接后发现所涉及到的知识点很繁杂，倒也非常正常。但如果你暂时先将它们放在一边——然后先去进行上一段中提到的“三步走”的路径，你会发现其实在整个过程中，对这些名词的陌生感并不会让你对解题的过程产生什么明显的障碍。

    当然，这样的观点是一种马后炮了，但是正如雷军曾经在年度演讲中所说：“知识不全是线性的，大部分是网状的，知识点之间不一定有绝对的先后关系；前面内容看不懂，跳过去，并不影响学后面的；后面的学会了，有时候更容易看懂前面的。”然后可能回过头你就会更容易发现，你在这个问题中需要的是什么——这是本题在题目描述中的第一个设计，增加额外阅读材料来增加理解门槛。

    思路既然相对直接，下一步就是要考虑一下应该如何使用LLM了。仔细阅读题目，题目明确说到 “训练或微调你的LLM” “如何在算力受限的前提下，选择合适的大模型与策略实现目标”综合看来，不论你是使用了我们平台提供的算力资源，又或者是使用自己的本地算力，首先可能最好先试试一些1.5B模型。可是如果这样操作，你会发现最终得到的训练结果不尽如人意——因为1.5B的模型并不是很好地适配着本次实验的精度要求，生成的行为树也会在内容和结构上较为混乱。

    那再增大到使用7B模型呢？或许此时本地算力资源已经有些窘迫，要是使用全精度，需要的显存应该已经有些超过了算力支撑，如果改为半精度或者量化后使用，对于本次任务的精确性要求又是否能够胜任呢？此时再想想后续的步骤——选用了LLM并且本地部署之后，还需要进行微调，所以所需要涉及的操作成本和不确定的结果，在有限的比赛时间内，让这个办法显得麻烦且充满不确定性。那怎么办呢？那此时你需要做的是跳出一些题面上暗示的方向，转而直接调用更大模型的API，它可以解决上述的几乎所有问题，只不过可能需要花一点点米而已（不过本来国内大模型很多都有免费额度，它们也基本可以让你顺利完成本任务，或cover大部分消耗了）——这是本题在题目描述中的第二个设计，对你的实操方向进行一定的故意误导。

    然后需要做的事情便很明确了。至于适配CPP库，Groot可视化等方面的要求，如果你是本地部署并训练后推理，那这两步动作应该是自然而然可以做到的。如果你是调用API，那在处理inputs.json的过程中调整一下prompt的内容，便也可以轻松解决解决的。所以本题并没有什么难度，对吗？

## 相关知识

### BehaviorTree.CPP 库适配

如果要让生成的 xml 格式行为树适配这个库，那需要具备的特征有如下一些：

- **根节点**必须是 `<BehaviorTree>`：

  ```
  <BehaviorTree ID="MainTree">
    <!-- 子节点定义 -->
  </BehaviorTree>
  ```

- **XML 中的标签名\*\***必须对应注册到 `BehaviorTreeFactory` 的节点类名，例如：

  ```
  <Sequence>       # 对应 C++ 的 BT::SequenceNode
    <Action_A/>    # 对应注册的 "Action_A" 类
    <Condition_B/> # 对应注册的 "Condition_B" 类
  </Sequence>
  ```

- **输入/输出端口**：通过 `ports` 传递数据，需在 XML 中明确定义，例如：

  ```
  <CalculateGoal x="10" y="{target_y}" output_key="goal"/>
  ```

- **变量引用**：使用 `{variable_name}` 语法读写黑板变量，例如：

  ```

  <Sequence>
    <SetBlackboard value="42" output_key="counter" /> # 写入
    <Action_CheckCounter counter="{counter}" />      # 读取
  </Sequence>
  ```

- **外部子树引用**：通过 `<SubTree>` 标签复用其他 XML 中定义的行为树，例如：

  ```
  <BehaviorTree ID="MainTree">
    <Sequence>
      <SubTree ID="SubTree_A"/>  # 引用子树
    </Sequence>
  </BehaviorTree>

  <BehaviorTree ID="SubTree_A">
    <!-- 子树具体实现 -->
  </BehaviorTree>
  ```

- **控制节点语法符合规则**，例如：

  ```
  # 顺序执行直到子节点失败
  <Sequence>
    <Child1/>
    <Child2/>
  </Sequence>

  # 选择第一个成功的子节点
  <Fallback>
    <ChildA/>
    <ChildB/>
  </Fallback>

  # 并行执行所有子节点
  <Parallel success_count="2" failure_count="1">
    <ChildX/>
    <ChildY/>
  </Parallel>
  ```

不过其实对于我们这个任务来说，上面的一些内容在本任务中其实也并没有涉及。

### Groot 可视化工具适配

这一个方面的适配就更容易一些，只需要做到：

- 根节点为 `<root>`，包含 `<TreeNodesModel>` 和 `<BehaviorTree>`
- 所有节点类型（包括内置节点）需在 `<TreeNodesModel>` 中声明
- 端口需明确定义名称和类型
- 子树引用需通过 `<SubTree>` 实现

相信你也发现了，这其中有很多和适配 BehaviorTree.CPP 库共通的地方。

## 解决过程与关键代码

这里我首先给出的是进行本地部署并微调的方法，使用的是 DeepSeek-R1-Distill-Qwen-7B 这个模型。如果你经过了类似题解开头所说的思考后，本题最合适的解决方式应该是调用模型的 API，毕竟现在我们并不是需要完全控制模型的推理过程，也不需要在离线环境下工作，同时这样的方式还可以突破算力资源的限制。

由于总的代码量不小（这个方法本来就比较麻烦），所以我在这里进行的展示只表现其中的关键部分。另外，我在题解中所涉及到的一些参数配置仅供参考，如果你研究出了能够让输出结果准确率更高的参数配置方案，那当然是更好了！

首先应该导入相关库并且将需要的参数配置好：

```
import json
import os
import torch
import numpy as np
import xml.dom.minidom
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset, DatasetDict
import pandas as pd
import re
import time
from sklearn.model_selection import train_test_split

!pip install -q transformers datasets peft accelerate bitsandbytes scikit-learn

CONFIG = {
    "model_name": "deepseek-ai/deepseek-r1-distill-qwen-7b",
    "model_cache_dir": "./model_cache",

    "input_file": "data_input.json",
    "output_file": "data_output.json",
    "context": "你将获得一个由行为树执行的任务摘要，你的目标是以XML格式表达这个行为树。",

    "output_dir": "./bt_generator_output",
    "epochs": 5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_length": 1024,
    "train_test_split": 0.8,

    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,

    "temperature": 0.5,
    "top_p": 0.7,
    "max_new_tokens": 1000,

    "save_path": "../bt_client/bt_xml/results.xml"
}

os.makedirs(CONFIG["model_cache_dir"], exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
```

配置都完成之后，就可以开始已进行模型的部署流程，这里需要先从互联网上获取模型文件。在部署过程中，我们进行了简单的 8 位量化以减少推理阶段所需的内存占用。来减少本地部署的实现方式下对资源的需求：

```
def deploy_model(force_download=False):
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"],
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inference_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=inference_config,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        device_map="auto",
        trust_remote_code=True
    )

    return base_model, tokenizer
```

此时模型的部署已经完成，只需要准备好数据，就可以准备开始动手了：

```
def prepare_data():
    with open(CONFIG["input_file"], 'r', encoding='utf-8') as f:
        inputs = json.load(f)
    with open(CONFIG["output_file"], 'r', encoding='utf-8') as f:
        outputs = json.load(f)

    data = []
    min_len = min(len(inputs), len(outputs))

    for i in range(min_len):
        prompt = f"<human>\n{CONFIG['context']}\n{inputs[i]}\n</human>\n<assistant>\n{outputs[i]}\n</assistant>"
        data.append({"text": prompt})

    train_indices, test_indices = train_test_split(
        range(len(data)),
        train_size=CONFIG["train_test_split"],
        random_state=42
    )

    train_dataset = Dataset.from_dict({
        "text": [data[i]["text"] for i in train_indices]
    })

    test_dataset = Dataset.from_dict({
        "text": [data[i]["text"] for i in test_indices]
    })

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    return dataset_dict

```

下一步是对模型进行微调，由于本任务是指令生成行为树，所以需要进行的类型是指令微调：

```
def prepare_model_for_training(base_model):
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, lora_config)

def tokenize_dataset(dataset_dict, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=CONFIG["max_length"],
            padding="max_length",
        )

    tokenized_datasets = DatasetDict({
        split: dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {split} set"
        )
        for split, dataset in dataset_dict.items()
    })


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    return tokenized_datasets, data_collator
```

微调也完毕了，此时训练就可以正式开始：

```
def train_model(model, tokenized_datasets, data_collator, tokenizer):
    training_args = TrainingArguments(
        output_dir=os.path.join(CONFIG["output_dir"], "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_ratio=CONFIG["warmup_ratio"],

        early_stopping_patience=3,
        load_best_model_at_end=True,

        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=20,
        save_steps=20,
        save_total_limit=3,

        logging_dir=os.path.join(CONFIG["output_dir"], "logs"),
        logging_steps=10,

        fp16=True,
        report_to="tensorboard",
    )

    early_stopping_callback = transformers.EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback]
    )

    trainer.train()

    final_model_path = os.path.join(CONFIG["output_dir"], "final_model")
    print(f"\n保存最终模型到 {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    return model, trainer
```

完成后使用我们已经在早先按照 8:2 的比例划分好的测试集来对训练情况进行测试，测试时可以设置 90%的要求，来让模型更可能达到拿到本题满分所需要的准确率。

如果验证结果达到了我们预设的要求，那下一步就可以正式将前面辛辛苦苦部署和训练的模型用于实践，将 inputs.json 交给它，让它进行推理和应用。这一步的代码中为了能够得到看到模型成功运行起来的一种成就感（当然其实还是为了看到程序运行完毕了），所以 print 了一些输出提示来显示模型运行的成功性：

```
def generate_behavior_tree(model, tokenizer, task_input):
    prompt = f"<human>\n{CONFIG['context']}\n{task_input}\n</human>\n<assistant>"
    model_input = tokenizer(prompt, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        result = tokenizer.decode(
            model.generate(
                **model_input,
                max_new_tokens=CONFIG["max_new_tokens"],
                temperature=CONFIG["temperature"],
                top_p=CONFIG["top_p"],
                do_sample=True
            )[0],
            skip_special_tokens=True
        )

        print(f"生成结果:")
        print(f"{result}")

    pattern = r'<root .*?</root>'
    matches = re.findall(pattern, result, re.DOTALL)

    if matches:
        final_tree = matches[-1]
        print("\n提取到的行为树:")
        print(final_tree)

        with open(CONFIG["save_path"], "w") as f:
            f.write(final_tree)
        print(f"行为树已保存到: {CONFIG['save_path']}")
        return final_tree
    else:
        print("未能提取到行为树XML")
        return None

def load_trained_model():
    final_model_path = os.path.join(CONFIG["output_dir"], "final_model")
    tokenizer = AutoTokenizer.from_pretrained(final_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        final_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("模型加载成功")

    return model, tokenizer
```

到此为止其实主要的逻辑都已经完成了，只需要将前面的各个功能块整合在一起就可以完成本题的任务。

当然你知道的，如果我们采取的是直接调用模型 API 的方式，那整个流程可以更加简化。你还可以试试当调用比较大参数量的模型 API 时，它其实甚至可以不需要经过前面的训练，而是直接去理解指令，就可以推理生成具备足够精确度、满足要求的行为树。

## 相关论文

- @inproceedings{Izzo_2024, title={BTGenBot: Behavior Tree Generation for Robotic Tasks with Lightweight LLMs},url={http://dx.doi.org/10.1109/IROS58592.2024.10802304}, DOI={10.1109/iros58592.2024.10802304}, booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, publisher={IEEE}, author={Izzo, Riccardo Andrea and Bardaro, Gianluca and Matteucci, Matteo}, year={2024}, month=oct, pages={9684–9690} }
- @inproceedings{Zhou_2024, title={LLM-BT: Performing Robotic Adaptive Tasks based on Large Language Models and Behavior Trees}, url={http://dx.doi.org/10.1109/ICRA57147.2024.10610183}, DOI={10.1109/icra57147.2024.10610183}, booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, publisher={IEEE}, author={Zhou, Haotian and Lin, Yunhan and Yan, Longwu and Zhu, Jihong and Min, Huasong}, year={2024}, month=may, pages={16655–16661} }
- @misc{ao2024behaviortreegenerationusing, title={Behavior Tree Generation using Large Language Models for Sequential Manipulation Planning with Human Instructions and Feedback}, author={Jicong Ao and Yansong Wu and Fan Wu and Sami Haddadin}, year={2024}, eprint={2409.09435}, archivePrefix={arXiv}, primaryClass={cs.RO}, url={https://arxiv.org/abs/2409.09435},}
- @misc{lykov2023llmbrainaidrivenfastgeneration, title={LLM-BRAIn: AI-driven Fast Generation of Robot Behaviour Tree based on Large Language Model}, author={Artem Lykov and Dzmitry Tsetserukou}, year={2023}, eprint={2305.19352}, archivePrefix={arXiv}, primaryClass={cs.RO}, url={https://arxiv.org/abs/2305.19352},}
