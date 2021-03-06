﻿## Graph Neural Networks for Social Recommendation

### ABSTRACT

问题：
但是，基于GNN构建社会推荐系统面临挑战。 例如，用户项图对交互及其关联的意见进行编码； 社会关系具有不同的优势； 用户涉及两个图（例如，用户-用户社交图和用户-项目图）。

提出的方法：
我们提供了一种原则性的方法来联合捕获用户项目图中的交互和观点，并提出了GraphRec框架，该框架连贯地对两个图和异构强度进行建模。


### INTRODUCTION

背景：
这些社交推荐系统是根据以下现象开发的：用户通常通过周围的人（例如，同学，朋友或同事）获取和传播信息，这意味着用户的基本社会关系可以在帮助他们过滤信息方面发挥重要作用。

(GNNs)已被提出来学习图形数据的有意义的表示,同时节点信息可以在变换和聚合后通过图传播

社会推荐数据的种类：
社会推荐中的数据可以表示为具有两个图形的图形数据。这两个图包括表示用户之间关系的社交图和表示用户与项目之间交互的用户项图。
由于这 两个图可以连接起来。此外，社交推荐的自然方法是将社交网络信息整合到用户和项目潜在因素学习中[37]。 学习项目和用户的表示形式是构建社交推荐系统的关键

基于GNN的社会推荐系统面临挑战。 
>社交推荐器系统中的社交图和用户项目图从不同的角度提供有关用户的信息。 重要的是，从两个图表中汇总信息以学习更好的用户表示。

1. 第一个挑战是如何固有地组合这两个图。 此外，用户项目图不仅包含用户与项目之间的交互，还包括用户对项目的意见。 例如，如图1所示，用户与“裤子”和“笔记本电脑”项目进行交互； 用户不喜欢“笔记本电脑”而喜欢“裤子”。

2. 第二个挑战是如何共同捕获用户与项目之间的交互和观点。

3.  第三个挑战是如何区分具有不同优势的社会关系。


### GraphRec 模型结构

**模型的整体结构图**

![GraphRec](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/model.png)
	

#### 2.1 Definitions and Notations

我们假设R∈Rn×m是用户项目评分矩阵，也称为用户项目图。 如果ui给vj评分，则rij为评分分数，否则我们用0表示从ui到vj的未知评分，即rij =0。观察到的评分分数rij可被视为用户ui对 项目vj。

令N（i）为ui直接连接的用户集合，C（i）为ui交互的项目集合，而B（j）为与vj交互的用户集合。 另外，用户可以彼此建立社会关系。 我们使用T∈Rn×n表示用户-用户社交图，如果uj与ui有关系，则Tij = 1，否则为零。 给定用户项目图R和社交图T，我们旨在预测R中的缺失评级值。


#### 2.2 An Overview of the Proposed Framework

该模型包含三个组件：用户建模，项目建模和评级预测。 
- 第一个组件：
	用户建模，它是要学习用户的潜在因素。 由于社交推荐系统中的数据包括两个不同的图，即社交图和用户项目图，因此我们有很大的机会从不同的角度学习用户表示。 因此，引入了两个聚合来分别处理这两个不同的图。
	- 一种是项目聚合，可以通过用户与用户项目图中的项目（或项目空间）之间的交互来了解用户。 

	- 另一个是社交聚合，即社交图中用户之间的关系，可以帮助从社交角度（或社交空间）建模用户。 直观地通过组合来自项目空间和社交空间的信息来获得用户潜在因素。 

- 第二个组件：
	项目建模，用于学习项目的潜在因素。 为了在用户项目图中同时考虑交互和观点，我们引入了用户聚合，即在项目建模中聚合用户的观点。 

- 第三个组件是通过集成用户和项目建模组件来通过预测学习模型参数。
   

### 2.3 User Modeling

问题：如何固有地将用户项目图和社交图结合起来。

方法：使用两种类型的聚合来从两个图中学习因子。

- 第一种聚合（表示为项目聚合）被用来从用户-那里获取项目空间用户潜在因子hIi∈Rd。 项聚合的目的是通过考虑用户ui与之交互的项以及用户对这些项的意见，来学习项空间用户潜在因子hIi。 
    
    为了数学上表示此聚合，我们使用以下函数作为：
	
	![公式1](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/math1.png)
	
	其中C（i）是用户ui已与之交互的项目集（或用户项目图中ui的邻居），
	
	xia是表示ui与项目va之间的感知感知交互的表示向量，而Aggreitems为 项目汇总功能。 
	
	
	![公式2](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/math4.png)

	其中xia由上图可得，对于这个$g_v$其实是两层神经网络，详情可参考代码。

	其中⊕表示两个向量之间的串联运算。
	
	Aggreitems的一种流行的聚集函数是均值算子，我们采用{xia，∀a∈C（i）}中向量的元素平均法。其函数如下： 
	![公式3](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/math2.png)
    
    为了缓解基于注意力机制启发的基于均值的聚合器的局限性，一种直观的解决方案是调整αi以了解目标用户ui，即为每个（va，ui）对分配个性化权重 ，
	![公式4](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/match3.png)
    
    其中αai固定为1 | C（i）| 基于均值的聚合器中的所有项目。
    其中αia表示与va互动的注意力权重，当根据互动历史记录C（i）来表征用户ui的偏好时，有助于用户ui的项目空间潜在因素。
    特别地，我们使用两层神经网络将项目注意αia参数化，我们将其称为注意网络。 注意网络的输入是交互和目标用户ui的pi的意见感知表示xia。
    正式地，注意力网络被定义为：
    
    ![公式5](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/math5.png)
    
    上式得到的α^{*}_{ia}通过SoftMax,得到归一化的最后的权重α_{ia}，就是公式4中的αia
    
    ![公式6](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/math6.png)
 
   
    
- 第二种聚合是社交聚合，其中从社交图了解社交空间用户潜在因子hSi∈Rd。 
    
    从这种社交角度表示用户潜在因素，我们提出了社交空间用户潜在因素，这是从社交图中聚合相邻用户的项目空间用户潜在因素。 特别地，ui的社交空间用户潜能因子hSi是聚合ui的邻居N（i）中用户的项目空间用户潜能因子，如下所示：
    
    ![公式7](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/math7.png)
    
    其中h_{0}^{I}是指user model中的输出结果，Aggre类似于项目聚集中的均值算子，它采用hIo中向量的元素式均值。公式如下:
     
     ![公式8](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/match8.png)
     
     其中βi固定为1 | N（i）| 基于均值的聚合器的所有邻居。 假设所有邻居均对用户ui的表示做出同等贡献
     
     因此我们通过两层神经网络执行注意力机制，以通过将社会注意力βio与hIo和目标用户嵌入pi关联起来，提取对ui有重要影响的这些用户，并建立他们的联系强度，如下所示：
     
     ![公式9-11](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/math911.png)
     

**Learning User Latent Factor**

将上面这两个因素组合在一起以形成最终用户潜在因素hi，然后将hi输入到MLP中。公式如下：

![公式9-11](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/math1214.png)

其中l是MLP网络的层数


### 2.4 Item Modeling

项目建模用于通过用户聚合来了解itemvj的项目潜在因子，表示为zj。

**用户聚合**： 同样，我们使用类似于通过项目聚合来学习项目空间用户潜在因素的方法。 对于每个项目vj，我们需要汇总来自与vj进行交互的一组用户的信息，记为B（j）。

内容：

1. 即使对于同一项目，用户在用户项目交互期间也可能表达不同的意见。 来自不同用户的这些意见可以以用户提供的不同方式捕获同一项目的特征，这可以帮助建模项目潜在因素。 对于从ut到vj与意见r的交互，我们引入了一个意见感知交互用户表示fjt，它是通过MLP从基本用户嵌入pt和意见嵌入er中获得的，表示为дu。 解决方案是将交互信息与意见信息融合在一起：
	
	![公式15](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/math15.png)

2. 为了学习项目潜在因子zj，我们还建议在B（j）中针对项目vj聚合用户的意见感知交互表示。 用户聚合函数表示为Agguser，用于将fjt中的用户感知感知的交互表示聚合为：

	![公式16](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/math16.png)

3. 我们引入了一种注意机制，以fjt和qj为输入，通过两层神经注意网络来区分用户的重要权重µjt

	![公式17-19](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/match1719.png)

用户关注度是为了捕获用户项目交互对学习项目潜在因素的不同影响。

### Rating Prediction

GraphRec模型应用于评级预测的推荐任务。 借助用户和项目的潜在因素（即hi和zj），我们可以先将它们hi⊕zj串联起来，然后将其输入到MLP中以进行评分预测：

![公式20-23](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/math2023.png)


### 2.6 Model Training

该模型采用平方损失函数，公式如下：
	
![公式24](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/match24.png)

 为了优化目标函数，我们在实现中采用RMSprop作为优化器，而不是普通的SGD。 每次，它会随机选择一个训练实例，并朝其负梯度的方向更新每个模型参数。 我们的模型中有3种嵌入，包括项目嵌入qj，用户嵌入pi和意见嵌入er。 它们是随机初始化的，并在培训阶段共同学习。 由于原始特征非常大且稀疏，因此我们不使用one-hot向量来表示每个用户和每个项目。 通过将高维稀疏特征嵌入到低维潜在空间中，可以轻松训练模型。同时为了防止过拟合，采用drop的方式进行训练，并在测试中关闭drop。


### Experiment

与一些baseline比较所得结果：

![表格3](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/table3.png)

后续有分别对社交网络与用户意见的影响做了分析：

![Figure3](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/f3.png)

对注意力机制的探索：

![Figure4](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/f4.png)

也对embedding的大小做了比较：

![Figure5](https://github.com/swallown1/NoteOfPaper/blob/master/WWW/GNNforSocialRec/images/f5.png)


### Conclusion

在本文中，我们提出了针对社会推荐问题的SocialGCN模型。 我们的模型结合了GCN的优势（用于对社交网络中的扩散过程进行建模）和经典的基于潜在因子的模型来捕获用户项目偏好。 具体来说，用户嵌入是按分层扩散方式构建的，初始用户嵌入取决于当前用户的特征以及用户特征向量中未包含的自由用户潜在向量。 同样，每个项目的潜在矢量也是该项目的免费潜在矢量及其特征表示的组合。 我们表明，当用户和项目属性不可用时，建议的SocialGCN模型是灵活的。 实验结果清楚地表明了我们提出的模型的灵活性和有效性。 例如，在Flickr上，SocialGCN比NDCG的最佳基准提高了13.73％。 将来，我们希望探索GCN，以用于更多的社会推荐应用，例如社会影响力建模，时间社会推荐等。