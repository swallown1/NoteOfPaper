## Session-based Recommendation with Graph Neural Networks

>论文链接：![SRGNN](../../paper/SR-GNN.pdf)
>代码实现：![SRGNN](https://github.com/CRIPAC-DIG/SR-GNN)

本论文的主要的创新点是将Session中的一些item序列转化成图的形式，再通过GRU利用t-1时刻节点向量学习节点t时刻的向量，通过session中的最后的节点向量表示session的向量表示，也是短期兴趣。再通过软注意力机制学习session的长期兴趣。最终将两者结合起来通过交叉熵损失进行优化。

### Abstract

基于会话的推荐目标是预测基于匿名会话的用户操作。

之前的方法是通过将会话建模为序列，并评估用户表示以用来做推荐。

之前方法的不足是基于会话的用户表示不够准确。为了获得准确的项目嵌入并考虑到项目的复杂转换。

1.会话序列被建模成图结构数据。
2.基于会话图，GNN可以捕获项目的复杂转换
3.使用注意力网络将每个会话表示为该会话的全局偏好和当前兴趣的组成。

### 1 Introduction

基于会话的推荐场景是用户的名片以及历史活动的非持续记载，只有在session中的记录。但是传统基于序列的推荐算法并不能获取items间的复杂的转移关系，因此本文提出了基于GNN的方法。


### 2 Related Work

 - 传统的推荐算法
 - 基于深度学习的方法
 - 图神经网络(GNN)

<b>传统方法的问题在于</b>

1. 在没有足够多的用户行为下很难估计用户表示。

2. 基于会话的推荐系统中，会话大多是匿名的，并且会话数众多，并且包含在会话点击中的用户行为通常受到限制

3. 常常忽略了远处项目之间的复杂过渡。

<b>提出的SR-GNN 模型包括以下四个步骤：</b>

- 会话图建模
- 节点表示学习
- 会话表示生成
- 生成推荐结果


### 3 The Proposed Method

<b>SR-GNN的整体框架如下图所示：</b>

![](https://github.com/swallown1/NoteOfPaper/blob/master/AAAI/Session_based_Rec_GNN/images/f1.png)


#### 3.1符号定义
在基于会话的推荐系统中，设V={v1,v2,…,vm}是所有session中的涉及的独立物品，那么匿名的会话序列根据时间排序可以表示为s=[vs,1,vs,2 ,…vs,n ]。其中，Vs,i表示用户在会话s中点击的物品。因此基于会话的推荐系统的目标就是预测用户的下一次点击，比如Vs,n+1。当然在基于会话的推荐系统中，对于某一会话s，系统一般给出输出概率最高的几个预测点击目标，作为推荐的候选。

#### 3.2 构造会话图

每一个会话序列s都可以被建模为一个有向图Gs=(Vs,Es)。在该会话图中，每个节点都代表一个物品vs,i ,每一条边(vs,i−1,vs,i)代表在会话s中，用户在点击了物品vs,i−1 后点击了vs,i 
​	
因为许多item可能会在会话序列中多次出现，因此**论文给每一条边赋予了标准化后的加权值，权重的计算方法为边的出现次数除以边起点的出度。**

基于每个节点的词嵌入向量的表示形式，每个会话s就可以嵌入向量表示：各个节点的词嵌入向量按时间顺序拼接而成。


####　3.3在会话图上的学习物品嵌入向量(Learning Item Embeddings on Session Graphs)

GNN十分适合用于基于会话的推荐算法，因为它可以根据丰富的节点连接自动提取会话图的特征。门控图神经网络(Gated GNN)的更新可以由如下公式给出：

![](https://github.com/swallown1/NoteOfPaper/blob/master/AAAI/Session_based_Rec_GNN/images/math15.png)

上式中的H控制着权重，$z_{s,i}, r_{s,i}$ 分别代表重置(reset)和更新(update)门。$[v_1^{t-1},...,v_n^{t-1}]$是会话s包含的节点向量。$ A_s \in R^{n \times 2n}$ 是关系矩阵,决定着图中的节点彼此间如何关联的。$ A_{s,i:} \in R^{1 \times 2n}$ 是As 中与节点$v_{s,i}$相关的两列(因为会话图是有向图，因此这的两列分别对应的是当前节点到其他节点和其他节点到当前节点对应的关系系数)。

第一个等式利用连接矩阵(connection matrix)从邻接节点中整合信息。其他等式的更新与GRU模型类似。

连接矩阵(connection matrix) $A_s \in R^{n \times 2n}$ 决定了图中的节点之间如何连接，$A_s$ 由两个邻接矩阵( $A_s^{(out)}$ 和 $A_s^{(in)}$ )拼接(concat)而成， $A_{s,i:}$表示的是节点 $v_{s,i}$ 分别在 $A_s^{(out)}$ 和 $A_s^{(in)}$ 对应的两列 

![](https://github.com/swallown1/NoteOfPaper/blob/master/AAAI/Session_based_Rec_GNN/images/f2.png)

对于上面的式子对于每个会话图Gs，门控图神经网络同时处理节点。 等式 （1）用于在矩阵As给出的限制下，在不同节点之间进行信息传播。 具体来说，它提取邻域的潜在向量并将它们作为输入输入到图神经网络。 然后，两个门，即更新和复位门，分别决定要保留和丢弃什么信息。 之后，我们根据方程式中的描述，通过先前状态，当前状态和复位门来构造候选状态。  （4）。 然后，最终状态是在更新门的控制下，先前隐藏状态和候选状态的组合。 在更新会话图中的所有节点直到收敛之后，我们可以获得最终的节点向量。


#### 3.4 Generating Session Embeddings

对于先前基于会话推荐，通常直接假设会话的Enbedding，而SR-GNN是从会话节点的中学习而来。而为了更好地预测用户的下次点击，我们计划制定一项策略，将长期偏好和会话的当前兴趣相结合，并使用这种组合的嵌入作为会话嵌入。


过将所有的会话图送入G-GNN中能够得到所有节点的嵌入向量。接下来为了将每个会话表示为嵌入向量s∈Rd 首先考虑局部嵌入向量s1,对于会话s=[vs,1 ,vs,2 ,…vs,n ],局部嵌入向量可以简单定义为会话中最后点击的物品vs,n,对于具体的session也可以简单表示为vn ,即s1=vn。

然后，论文结合所有节点嵌入向量来计算会话图的全局嵌入向量sg
​,鉴于不同节点信息可能存在不同的优先级，为了使全局嵌入向量有更好的表现，论文引入了soft-attention机制。
 
![](https://github.com/swallown1/NoteOfPaper/blob/master/AAAI/Session_based_Rec_GNN/images/math6.png)

其中参数vn表示的就是会话s的局部嵌入向量，vi就是会话中的各个节点，参数$q \in R^d$, $W_1,W_2 \in R^{d \times d}$均是可训练的，随训练迭代，控制着每个节点嵌入向量的权重$ \alpha_i$，最后对于每个节点对应的词嵌入向量进行加权求和得到最后的全局词嵌入向量。

最后将会话的局部嵌入向量和全局嵌入向量相结合即可得到融合的嵌入向量
$$s_h=W_3[s_1;s_g]$$

#### Making Recommendation and Model Training

得到每个会话的嵌入向量后论文对每个候选物品vi计算得分$\hat{z_i}$,具体计算公式如下：

![](https://github.com/swallown1/NoteOfPaper/blob/master/AAAI/Session_based_Rec_GNN/images/math8.png)

然后经过Softmax激活函数后得到模型的预测输出：
$$\hat y = Softmax(\hat Z)$$

对于每个会话图，损失函数选用常见的交叉熵函数：

![](https://github.com/swallown1/NoteOfPaper/blob/master/AAAI/Session_based_Rec_GNN/images/math10.png)

模型的训练采用基于时间的方向传播方法(BPTT)，由于在基于会话的推荐方案中，大多数会话长度较短，因此训练迭代次数不要太多，防止过拟合。

## 参考资料

[论文笔记」Session-based Recommendation with GNN](https://zhuanlan.zhihu.com/p/82796415)

[SR-GNN论文解读(AAAI2019)](https://blog.csdn.net/yfreedomliTHU/article/details/91345348)

[模型解读：基于会话的最优推荐模型SR-GNN](https://zhuanlan.zhihu.com/p/65749652)

