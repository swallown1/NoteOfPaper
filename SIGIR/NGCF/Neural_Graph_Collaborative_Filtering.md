## Neural Graph Collaborative Filtering  (图神经协同过滤)


>论文链接：[Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108?context=cs.IR)
>
>代码实现:[code](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
>
>原理：在user-item interaction graph 上使用GCN 来学习高阶连通中的协同因子，将其嵌入到user和item的向量表示上，然后模型通过内积的方式进行相互建模。
>
>与其他使用GNN处理的区别：
>
>1. 大部分使用GNN进行处理的论文，只学习User向量，这篇中item也是使用GNN进行学习
>
>2. 大部分论文是在知识图谱KG或者社交网络Social Network上使用GNN，这篇论文是在用户-项交互二部图上使用GNN

在讲NGCF之前，先说说相关概念

#### CF中关键方法：

- 嵌入: 将user 和 item转化为矢量化表示

- 交互建模：重建历史嵌入的模型，例如，矩阵因式分解(MF)直接嵌入用户/项ID作为向量，并建立用户-项与内积交互的模。

>#### 背景:
>
>- 协同过滤学习模型从项目的丰富侧面信息中学习的深层表示来扩展MF嵌入功能；
>- 神经协同过滤模型则用非线性神经网络取代内积的MF交互函数。
>
>以上的方法在交互建模中都是有效的，但是却无法为CF提供更好的嵌入表示，因为缺少一个协同信号（它潜伏在用户与项目之间的交互>中，以揭示用户(或项目)之间的行为相似性。）更具体的是，现有的大多数方法只使用描述性的特性(例如id和属性)构建嵌入功能，而不考虑用户-项的交互-这些功能只用于定义模型训练的目标函数。因此必须依赖交互功能弥补次优嵌入的不足。虽然直接地使用交互功能能弥补次优嵌入的不足，但是在实际开发中，交互的规模可以容易地达到数百万甚至更大，使得难以提取期望的协作信号。所以在这项工作中，我们解决了所有问题，通过利用来自用户项目交互的高阶连通性，在交互图形结构中对协作信号进行编码的方式。

#### 高阶连通性

![figure 1](https://github.com/swallown1/NoteOfPaper/blob/master/SIGIR/NGCF/image/figure1.png)

图1左边所示的为协同过滤用户-项目交互的基本交互图，双圆圈表示需要预测的用户u1，右图为左图以用户u1为根节点扩展的树形结构，l为到达用户u1的路径长度（可以作为兴趣重要度的权重值）
从右图中可以看到，同路径长度为3的项目i4、i5中，明显用户对i4的兴趣度高于i5，这是因为<i4,u1>连接的路径有两条，分别为i4->u2->i2->u1、i4->u3->i3->u1，而<i5,u1>则只有一条，为i5->u2->i2->u1。所以通过这些树形结构来查看u1对项目的兴趣，看项目与用户的连通性。这就是高阶连通性的概念。

#### 优化点：

- 在最近的名为Hop-Rec的方法中已经考虑了高阶连通性信息，但是它只是用于丰富训练数据，具体来说HOPRec的预测模型仍然是MF，而它是通过优化具有高阶连通性的亏损来训练的。

- 而NGCF则不同，这是一种将高阶连通性集成到预测模型中的新技术，该技术在经验上比Cf的HOP-Rec具有更好的嵌入效果。


### NGCF模型 

#### 1. 摘要

本文作者认为现有的工作通常通过从描述用户(或项目)的现有特性(如ID和属性)映射来获得用户(或项目)的嵌入。但是这种方法的一个固有缺点是，隐藏在**用户-项目交互中的协作信号**没有在嵌入过程中编码。因此，由此产生的嵌入可能不足以捕获协作过滤效果。

而在这项工作中，将用户-项目交互更具体地集成到嵌入过程中**二部图结构**。提出了一种新的推荐框架神经图协同过滤算法(NGCF)，**该算法利用用户项图的结构，在用户项图上传播嵌入**。这就导致了用户项图中高阶连通性的表达建模，有效地将协作信号显式地注入到嵌入过程中。


### 2.模型结构

**模型主要包括以下结构：**

1. 嵌入层：给模型提供初始的嵌入向量，包括用户嵌入和项嵌入。

2. 嵌入传播层：通过多个嵌入传播层学习高阶连通性中的协同因子，编码进用户和项的表示。

3. 预测层：将来自不同传播层的精化嵌入集合起来，并通过内积模拟交互，输出用户-项对的亲和力分数。


#### 2.1嵌入层
初始化用户和项目交互的数据,建立参数矩阵.
将用户和相互初始化为以下格式：

![embedding](https://github.com/swallown1/NoteOfPaper/blob/master/SIGIR/NGCF/image/embedding.jpg)

**优化点：**

这种端到端的优化，传统的神经协同和MF都是直接输入数据到交互层(或操作符)以达到预测分数。相反，在NGCF框架中，通过在用户项交互图上传播嵌入来改进嵌入。因为嵌入细化步骤明确地将协同信号注入到嵌入中。

####  2.2嵌入传播层：

接下来这幅图就是NGCF模型，通过模型捕获协同因子编码进嵌入表示中。

![NGCF](https://github.com/swallown1/NoteOfPaper/blob/master/SIGIR/NGCF/image/NGCF.jpg)

其中箭头表示的是信息流，通过多层嵌入传播对user1和item4 的嵌入进行优化，最终将所有嵌入表示联合已进行最终的预测。

##### 2.21 First-order Propagation

**消息构建(Message Construction)：**
	$$m_{u<-i} = f(e_i,e_u,P_{ui})$$

其中Pui是折损因子，就是信息传播在随路径长度而衰减，其中Mu←i是消息嵌入(即要传播的信息)。F(·)是消息编码函数，它以嵌入Ei和eu作为输入，并使用系数Pui进行控制l边(u，i)上每个传播的衰减因子。在这项工作中，我们将f(·)实现为：

![消息构建公式](https://github.com/swallown1/NoteOfPaper/blob/master/SIGIR/NGCF/image/gongsi1.png)

其中W1，W2∈Rd’×d为可训练权矩阵，提取有用信息进行传播，d '为转换大小。
Nu，Ni 表示用户 u 和项目 i 的第一跳邻居，
括号前面的是系数是拉普拉斯标准化，从表示学习的角度，反映了历史项目对用户偏好的贡献程度。从消息传递的角度来看，可以解释为折扣因子，因为所传播的消息应该随着路径长度而衰减。
在考虑信息嵌入时，不是只考虑了项目的影响，而且将 ei 和 eu 之间的相互作用额外编码到通过 ei⊙eu 传递的消息中。这使得消息依赖于 ei 和 eu 之间的亲和力，例如，从相似的项传递更多的消息。

**消息聚合(Message Aggregation):**

此阶段就是基于上一步，各节点在信息构建后，整合从u的领域传播的消息，用来改进u的表示(对于项也是一样的):

![消息聚合公式](https://github.com/swallown1/NoteOfPaper/blob/master/SIGIR/NGCF/image/xxjh.png)

e(1)u 是用户 u 在一阶嵌入传播层获得的表示
除了从邻居 Nu 传播的消息外，还考虑了 u 的自连接:mu←u = W1eu，保留了原始特征的信息。
类似地，我们可以通过从其连接的用户传播信息来获得项目 i 的表示形式 e(1)i。


##### 2.22 High-order Propagation
通过堆叠 l 嵌入传播层，用户(和项)能够接收从其 l-hop 邻居传播的消息。在第 l 步中，用户 u 的表示递归式为:

![高阶传播公式](https://github.com/swallown1/NoteOfPaper/blob/master/SIGIR/NGCF/image/gjcb.png)

其中也包括着对自己信息的传播

##### 2.23 矩阵计算形式

为了方便批量实现，提供了分层传播规则的矩阵形式:

![](https://github.com/swallown1/NoteOfPaper/blob/master/SIGIR/NGCF/image/jz1.png)

E(l) ∈ R(N+M)×dl 是用户和项经过 l 步嵌入传播后得到的表示
I 表示一个单位矩阵
L 表示用户-项目图的拉普拉斯矩阵:

![](https://github.com/swallown1/NoteOfPaper/blob/master/SIGIR/NGCF/image/jz2.png)

R ∈ RN×M 为用户-项目交互矩阵
0 为全 0 矩阵;
A 为邻接矩阵，D 为对角度矩阵，其中第 t 个对角元素 Dtt = |Nt |，这样 Lui 就等于之前的系数 pui

####  2.3模型预测：

由于在不同层中获得的表示强调通过不同连接传递的消息，所以它们在反映用户偏好方面有不同的贡献。
因此，将它们串联起来，构成用户的最终嵌入；对 item 也做同样的操作。

![](https://github.com/swallown1/NoteOfPaper/blob/master/SIGIR/NGCF/image/mxyc.png)

其中||为串联操作。除了连接，其他聚合器也可以应用，如加权平均、最大池、LSTM。使用串联在于它的简单性，不需要学习额外的参数，而且已经被非常有效地证明了。
最后，我们进行内积来估计用户对目标物品的偏好:

![](https://github.com/swallown1/NoteOfPaper/blob/master/SIGIR/NGCF/image/mxyc2.png)

####  2.4 Optimization:

该模型采用成对BPR损失
考虑观察到的和未观察到的用户-项目交互之间的相对顺序。具体地说，BPR 假设用户引用的已观察到的交互作用应该比未观察到的交互作用具有更高的预测值。目标函数如下：

![](https://github.com/swallown1/NoteOfPaper/blob/master/SIGIR/NGCF/image/opt.png)