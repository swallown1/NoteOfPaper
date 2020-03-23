## Wide&Deep

---

之前大致看过这篇论文，但是没有仔细的看，最近在弄一个二手车价格预测的小比赛学习，突然想到这个模型，于是回过头来仔细看了一遍。


## 一、Motivation
推荐系统的主要挑战之一，是同时解决Memorization和Generalization，理解这两个概念是理解全文思路的关键，下面分别进行解释。

### Memorization
面对拥有大规模离散sparse特征的CTR预估问题时，将特征进行非线性转换，然后再使用线性模型是在业界非常普遍的做法，最流行的即「LR+特征叉乘」。Memorization 通过一系列人工的特征叉乘（cross-product）来构造这些非线性特征，捕捉sparse特征之间的高阶相关性，即“记忆” 历史数据中曾共同出现过的特征对。

例如，特征1——专业: {计算机、人文、其他}，特征2——下载过音乐《消愁》:{是、否}，这两个特征one-hot后的特征维度分别为3维与2维，对应的叉乘结果是特征3——专业☓下载过音乐《消愁》: {计算机∧是，计算机∧否，人文∧是，人文∧否，其他∧是，其他∧否}。

典型代表是LR模型，使用大量的原始sparse特征和叉乘特征作为输入，很多原始的dense特征通常也会被分桶离散化构造为sparse特征。这种做法的优点是模型可解释高，实现快速高效，特征重要度易于分析，在工业界已被证明是很有效的。Memorization的缺点是：

需要更多的人工设计；
可能出现过拟合。可以这样理解：如果将所有特征叉乘起来，那么几乎相当于纯粹记住每个训练样本，这个极端情况是最细粒度的叉乘，我们可以通过构造更粗粒度的特征叉乘来增强泛化性；
无法捕捉训练数据中未曾出现过的特征对。例如上面的例子中，如果每个专业的人都没有下载过《消愁》，那么这两个特征共同出现的频次是0，模型训练后的对应权重也将是0；


#### Generalization
Generalization 为sparse特征学习低维的dense embeddings 来捕获特征相关性，学习到的embeddings 本身带有一定的语义信息。可以联想到NLP中的词向量，不同词的词向量有相关性，因此文中也称Generalization是基于相关性之间的传递。这类模型的代表是DNN和FM。

Generalization的优点是更少的人工参与，对历史上没有出现的特征组合有更好的泛化性 。但在推荐系统中，当user-item matrix非常稀疏时，例如有和独特爱好的users以及很小众的items，NN很难为users和items学习到有效的embedding。这种情况下，大部分user-item应该是没有关联的，但dense embedding 的方法还是可以得到对所有 user-item pair 的非零预测，因此导致 over-generalize并推荐不怎么相关的物品。此时Memorization就展示了优势，它可以“记住”这些特殊的特征组合。

Memorization根据历史行为数据，产生的推荐通常和用户已有行为的物品直接相关的物品。而Generalization会学习新的特征组合，提高推荐物品的多样性。 论文作者结合两者的优点，提出了一个新的学习算法——Wide & Deep Learning，其中Wide & Deep分别对应Memorization & Generalization。

## 二、Model

 - 模型结构

Wide & Deep模型结合了LR和DNN，其框架图如下所示。


Wide 该部分是广义线性模型，即

$y = W^T [x,f(x)]+b$，其中 x 和 f(x) 表示原始特征与叉乘特征。

Deep 该部分是前馈神经网络，网络会对一些sparse特征（如ID类特征）学习一个低维的dense embeddings（维度量级通常在O(10)到O(100)之间），然后和一些原始dense特征一起作为网络的输入。每一层隐层计算：

$a^{l+1}=f(W^l a^l+ b^l)$ ，其中 a^l,b^l,W^l 是第 i 层的激活值、偏置和权重， f 是激活函数。

损失函数 模型选取logistic loss作为损失函数，此时Wide & Deep最后的预测输出为：

$$
p(y=1|x) = \sigmod(W^T_{wide}[x,f(x)]+W^T_{deep} a^{lf}+b)
$$

其中 $\sigmod$表示sigmoid函数， f(x) 表示叉乘特征， $a^{lf}$ 表示NN最后一层激活值。


 - 联合训练

联合训练（Joint Training）和集成（Ensemble）是不同的，集成是每个模型单独训练，再将模型的结果汇合。相比联合训练，集成的每个独立模型都得学得足够好才有利于随后的汇合，因此每个模型的model size也相对更大。而联合训练的wide部分只需要作一小部分的特征叉乘来弥补deep部分的不足，不需要 一个full-size 的wide 模型。

在论文中，作者通过梯度的反向传播，使用 mini-batch stochastic optimization 训练参数，并对wide部分使用带L1正则的Follow- the-regularized-leader (FTRL) 算法，对deep部分使用 AdaGrad算法。