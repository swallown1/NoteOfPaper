## Session-based Recommendation with Graph Neural Networks

### Abstract

基于会话的推荐目标是预测基于匿名会话的用户操作。

之前的方法是通过将会话建模为序列，并评估用户表示以用来做推荐。

之前方法的不足是基于会话的用户表示不够准确。为了获得准确的项目嵌入并考虑到项目的复杂转换。

1.会话序列被建模成图结构数据。
2.基于会话图，GNN可以捕获项目的复杂转换
3.使用注意力网络将每个会话表示为该会话的全局偏好和当前兴趣的组成。

### Introduction

基于会话的推荐场景是用户的名片以及历史活动的非持续记载，只有在session中的记录。

![](https://github.com/swallown1/NoteOfPaper/blob/master/AAAI/Session_based_Rec_GNN/images/f1.png)

传统方法的问题在于
1. 在没有足够多的用户行为下很难估计用户表示。

2. 基于会话的推荐系统中，会话大多是匿名的，并且会话数众多，并且包含在会话点击中的用户行为通常受到限制
3. 常常忽略了远处项目之间的复杂过渡。

提出的SR-GNN 模型包括以下四个步骤：

- 会话图建模
- 节点表示学习
- 会话表示生成
- 生成推荐结果


### 3 The Proposed Method

####　3.1符号定义
在基于会话的推荐系统中，设V={v1,v2,…,vm}是所有session中的涉及的独立物品，那么匿名的会话序列根据时间排序可以表示为s=[vs,1,vs,2 ,…vs,n ]。其中，Vs,i表示用户在会话s中点击的物品。因此基于会话的推荐系统的目标就是预测用户的下一次点击，比如Vs,n+1。当然在基于会话的推荐系统中，对于某一会话s，系统一般给出输出概率最高的几个预测点击目标，作为推荐的候选。

####　3.2 构造会话图

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
