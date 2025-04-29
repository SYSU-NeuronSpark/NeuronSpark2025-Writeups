# NS-2025-00题解 —— 须弥识花大赛

题解作者：[HarveyMo](https://github.com/Master7Sword)

## 背景

如果要问什么是计算机视觉中最经典的问题，图像分类一定是被最先想到的那个。本题作为本次比赛的第一题以及签到题，选取了各位同学在人工智能实验课程中会涉及的图像分类作为开胃小菜。本题采用Oxford-Flowers-102细粒度花卉分类作为数据集，随机打乱后选取6150张图片作为训练集，2041张图片作为测试集。有关该数据集的具体信息以及各模型在该数据集上的表现可参考[Papers With Code](https://paperswithcode.com/sota/fine-grained-image-classification-on-oxford)。

## 解题思路

经过十几年的发展，深度学习在图像分类任务上已经衍生出了数不清的经典模型，比如ResNet[1], EfficientNet[2], MobileNet[3，4，5]等。相信你直接将本题投喂给大语言模型也会得到类似的推荐。这些经典的分类模型只要参数量不要太小（大于2M），基本上都有在本题中做到满分的潜力。

除了选择合适的模型外，你可能还需要调得一手好参，比如：
- resize尺寸：为了模型能够处理所有的输入，你肯定需要先将所有的输入resize成同一个大小，学界普遍采用的是224x224，适当增加这个尺寸或许可以减少图像信息的损失，从而增加模型判断的准确率。当然，代价是计算量随尺寸的平方成比例增长。
- 归一化：使输入数据分布一致，提升训练稳定性，同时可以对抗过拟合。
- 随机翻转/旋转：提升模型泛化能力的常用手段
- batchsize：不同任务最佳的batchsize不尽相同，需要多次试验才能确定
- 优化器：一般默认使用Adam，比其他优化器更通用
- 动态调整学习率：通过学习率调度器，精细调整模型参数，显著提高最终精度

作者本人也对这些超参数在不同的模型上进行了测试，具体结果见下方表格。

| 模型                     | 参数量  | resize尺寸       | normalize | 随机翻转/旋转 | batchsize | 优化器               | 学习率调度器          | Train Acc | Max Test Acc |
|--------------------------|---------|------------------|---------------|--------------------|-----------|----------------------|-----------------------|-----------|--------------|
| ResNet18 pretrained      | 11.7M   | 224              | -             | -                  | 32        | Adam                 | -                     | 99.41%    | 93.77%       |
| ResNet18 pretrained      | 11.7M   | 224              | 是     | -                  | 32        | Adam                 | -                     | 99.97%    | 90.83%       |
| ResNet18 pretrained      | 11.7M   | 224              | 是     | 是 | 32        | Adam                 | -                     | 99.56%    | 99.50%       |
| ResNet18 pretrained      | 11.7M   | 224              | -             | 是 | 32        | Adam                 | -                     | 99.64%    | 94.66%       |
| ResNet18 pretrained      | 11.7M   | 224              | 是     | 是 | 32        | Adam                 | ReduceLROnPlateau     | 100.00%   | 95.98%       |
| ResNet18 pretrained      | 11.7M   | 500              | -             | 是 | 32        | Adam                 | ReduceLROnPlateau     | 99.83%    | 98.33%       |
| MobileNetV2 pretrained   | 2.5M    | 224              | 是     | 是 | 32        | Adam                 | -                     | 100.00%   | 97.06%       |
| MobileNetV2 pretrained   | 2.5M    | 500              | -             | 是 | 32        | Adam                 | ReduceLROnPlateau     | 98.68%    | 98.68%       |
| MobileNetV3Large pretrained | 4.3M | 224              | 是     | 是 | 32        | Adam                 | ReduceLROnPlateau     | 99.97%         | 98.14%      |
| EfficientNet-B0 pretrained | 4.1M   | 500        | 是        | 是              | 32        | Adam   | ReduceLROnPlateau     | 98.77%   | 98.77%    | 78     |
| EfficientNet-B1 pretrained | 6.6M   | 500        | 是        | 是              | 32        | Adam   | ReduceLROnPlateau     | 98.92%   | 98.92%    | 53     |
| EfficientNet-B2 pretrained | 7.8M   | 500        | 是        | 是              | 32        | Adam   | ReduceLROnPlateau     | 99.02%   | 99.02%    | 93     |
| EfficientNet-B3 pretrained | 10.8M  | 500        | 是        | 是              | 16        | Adam   | ReduceLROnPlateau     | 99.26%   | 99.26%    | 32     |
| EfficientNet-B4 pretrained | 17.7M  | 500        | 是        | 是              | 16        | Adam   | ReduceLROnPlateau     | 99.31%   | 99.31%    | 100    |


## Reference

1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.
2. Mingxing Tan and Quoc V. Le. EfficientNet: Rethinking model scaling for convolutional neural networks. In ICML, 2019.
3. Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. MobileNets: Efficient convolutional neural networks for mobile vision applications. In CVPR, 2017.
4. Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. MobileNetV2: Inverted residuals and linear bottlenecks. In CVPR, 2018.
5. Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. Searching for MobileNetV3. In ICCV, 2019.