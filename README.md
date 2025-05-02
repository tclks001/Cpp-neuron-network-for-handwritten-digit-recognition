不使用任何第三方库，仅使用`C++`语言实现神经网络算法进行手写数字识别。

## 所需功能

-   矩阵运算单元 `matrix.cpp`
    矩阵类模板 `Matrix<T>`，支持矩阵的加、乘、数乘、转置等运算

-   神经网络层 `layer.cpp`
    神经网络层类`Layer`，有以下核心变量：

    -   `Matrix<T> W`：权重矩阵，$W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$
    -   `Matrix<T> b`：偏置矩阵，$b^{(l)} \in \mathbb{R}^{n_l \times 1}$
    -   `Matrix<T> z`：线性变换后的结果矩阵，$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} \in \mathbb{R}^{n_l \times B}$
    -   `Matrix<T> a`：激活函数后的结果矩阵，$a^{(l)} = \sigma(z^{(l)}) \in \mathbb{R}^{n_l \times B}$
    -   `string activate`：激活函数，支持`"sigmoid"`，`"ReLU"`，`"softmax"`等
        （其中第一层的输入 $a^{(0)} \in \mathbb{R}^{n_0 \times B}$ 为输入矩阵 $X$ ，最后一层的输出 $a^{(L)} \in \mathbb{R}^{n_L \times B}$ 为输出矩阵 $\hat{y}$ ）

-   神经网络模型 `model.cpp`
    神经网络模型类`Model`，有以下核心变量：

    -   `vector<Layer> layers`：神经网络层的集合
    -   `int input_size`：输入层的大小

## 训练过程

### 网络结构

输入层维度为 $n_0=28 \times 28=784$ （MNIST），输出层维度为$n_L=10$，激活函数为`"softmax"`。

### 初始化

随机初始化每层的权重矩阵 $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ 和偏置矩阵 $b^{(l)} \in \mathbb{R}^{n_l \times 1}$

-   对于`sigmoid`等对称激活函数，权重矩阵适合使用**Xavier**初始化，即权重满足均值为 $\mu=0$，方差为 $\sigma^2=\frac{2}{n_{l-1}+n_l}$ 的正态分布。偏置矩阵初始化为 $0$。
-   对于`ReLU`等非对称激活函数，权重矩阵适合使用**He**初始化，即权重满足均值为 $\mu=0$，方差为 $\sigma^2=\frac{2}{n_{l-1}}$ 的正态分布。偏置矩阵初始化为 $0$。

```CPP
#include <random>
std::random_device rd; // 随机数生成器
std::mt19937 gen(rd()); // Mersenne Twister 随机数生成器

std::normal_distribution<double> dist(mean, stddev);// 给定均值和标准差

double random_num = dist(gen);// 生成正态分布随机数
```

### Mini-batch 训练循环

共有 $m$ 个样本，训练数据为$X \in \mathbb{R}^{n_0 \times m}$，标签为$y \in \mathbb{R}^{n_L \times m}$（采用 one-hot 编码）。
每一轮 epoch，将数据打乱后分为若干个 batch （记得同时打乱 $X$ 和 $y$）
每批 $B$ 个样本，共有 $\lceil \frac{m}{B} \rceil$ 个 batch。
对于每个 batch，按照以下步骤进行训练：

1. 计算前向传播（Forward Propagation）：
    - 输入层：
      $$a^{(0)} = X \in \mathbb{R}^{n_0 \times B}$$
    - 隐藏层：
      $$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} \in \mathbb{R}^{n_l \times B}$$
      $$a^{(l)} = \sigma(z^{(l)}) \in \mathbb{R}^{n_l \times B}$$
      （其中 $\sigma$ 为激活函数，`"sigmoid"` 或 `"ReLU"`）
    - 输出层：
      $$z^{(L)} = W^{(L)} a^{(L-1)} + b^{(L)} \in \mathbb{R}^{n_L \times B}$$
      $$a^{(L)} = \text{softmax}(z^{(L)}) \in \mathbb{R}^{n_L \times B}$$
      （对一个矩阵执行激活函数，相当于对每个元素执行激活函数）
2. 计算损失函数（Loss）：
    - 使用**交叉熵损失函数**（Cross-Entropy Loss）：
        $$
        L = - \frac{1}{B} \sum_{i=1}^B \sum_{j=1}^{n_L} y_j^{(i)} \log a_j^{[L](i)}
        $$
        这是一个标量，表示这个 batch 中所有样本的平均损失。
3. 计算反向传播（Back Propagation）：
    - 在`"softmax"`激活函数下，输出层的梯度为：
        $$
        dZ^{(L)} = a^{(L)} - y \in \mathbb{R}^{n_L \times B}
        $$
    - 从后往前算：
        $$
        \frac{\partial L}{\partial W^{(l)}} = \frac{1}{B} dZ^{(l)} (a^{(l-1)})^T \in \mathbb{R}^{n_l \times n_{l-1}}
        $$
        $$
        \frac{\partial L}{\partial b^{(l)}} = \frac{1}{B} \sum_{i=1}^B dZ^{(l)} \in \mathbb{R}^{n_l \times 1}
        $$
4. 梯度下降更新参数（Gradient Descent）：
    - 对每层，给定学习率 $\alpha$，更新：
        $$
        W^{(l)}=W^{(l)}-\alpha \frac{\partial L}{\partial W^{(l)}}
        $$
        $$
        b^{(l)}=b^{(l)}-\alpha \frac{\partial L}{\partial b^{(l)}}
        $$
5. 重复以上步骤，直到达到最大迭代次数。
