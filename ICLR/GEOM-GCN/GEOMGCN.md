# GEOM-GCN: GEOMETRIC GRAPH CONVOLUTIONAL NETWORKS

- 论文 ：  https://arxiv.org/pdf/2002.05287.pdf
- 代码 ：  https://github.com/graphdml-uiuc-jlu/geom-gcn
- 来源 ：  ICLR 2020

###  摘要：
---
MPNN聚合器的两个基本弱点：
-  丢失邻域中节点的结构信息
-   缺乏捕获散布图中的长期依赖关系的能力。

本文提出了一种新颖的几何聚合方案，以克服这两个缺点。 背后的基本思想是，图上的聚合可以受益于图下方的连续空间。 提出的聚合方案由三个模块组成：节点嵌入，结构邻域和双层聚合。

### INTRODUCTION
----
消息传递神经网络（MPNN），例如GNN，GCN具有强大的学习能力。

尽管现有的MPNN已成功应用于各种场景，但MPNN聚合器的两个基本弱点限制了它们表示图结构数据的能力。 
1.  首先，聚合器丢失邻域中节点的结构信息。
        排列不变性是任何图学习方法的基本要求。 为了满足此要求，现有的MPNN采用排列不变的聚合函数(求和，最大值，求平均)，该函数将邻域中的所有“消息”视为一个集合。但是**这种聚合会丢失附近节点的结构信息，因为它没有区分来自不同节点的“消息”。**

2.其次，聚集器缺乏捕获散布图中的长期依赖关系的能力。
          在MPNN中，邻域定义为一个跳远的所有邻居（例如GCN）或直到r跳远的所有邻居（例如ChebNet）的集合。换句话说，具有这种聚合的MPNN倾向于学习图中近端节点的相似表示。这意味着它们可能是节点同形成立的分类图和社区网络的理想方法。

解决此限制的一种简单策略是使用多层体系结构，以便从远方节点接收“消息”。在CNN中采用的分层方式连接的多层来学习复杂的全局特征表示。但是MPNN无法进行深度网络，因为数据稀疏，如果层数过多会导致过于平滑。

**多层MPNN很难学习散布图的良好表示。**
-  一方面，来自远端节点的相关消息与多层MPNN中来自近端节点的大量无关消息无法区分地混合在一起，这意味着相关信息将被“冲走”并且无法有效地提取。
-   另一方面，在多层MPNN中，不同节点的表示将变得非常相似，并且每个节点的表示实际上都携带有关整个图的信息。

在本文中，我们从两个基本观察出发克服了图神经网络的上述缺点：
-  i）由于连续空间中的平稳性，局部性和组成性，经典神经网络有效地解决了类似的局限性。
-  ii）网络几何的概念弥合了连续空间和图之间的鸿沟。

网络几何旨在通过揭示网络下面的潜在连续空间来理解网络，该网络假定节点是从潜在连续空间中离散采样的，并且根据它们之间的距离来建立边缘。在潜在空间中，可以保留图中复杂的拓扑模式并将其呈现为直观的几何图形。

> 图上的聚合是否可以受益于连续的潜在空间，例如在空间中使用几何图形来构建结构邻域并捕获图中的长期依存关系？

从这个问题出发本文提出一个新的神经网络聚合方案，在该方案中，我们通过**节点嵌入将图映射到连续的潜在空间，然后使用在潜在空间中定义的几何关系来构建用于聚合的结构邻域**，然后设计一个双层离合器运行在结构领域以更新节点的特征表示。
**该方案提取了更多的图结构信息并可以聚合特征通过将遥远节点映射到潜在空间中定义的邻域来表示**

**贡献**：
i）针对图神经网络提出了一种新颖的几何聚合方案，该方案既可在图空间又可在潜伏空间工作，以克服上述两个缺点。
 ii）我们提出了一种用于图的跨语言学习的方案GEOMETRIC 的实现； 

### GEOMETRIC AGGREGATION SCHEME
GEOMETRIC 模型的几何聚合方案有三部分组成，具体如图所示：
![](https://upload-images.jianshu.io/upload_images/3426235-5652a8f98b38938e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上图中 A1-A2原始图被映射到一个潜在的连续空间。 B1-B2结构邻域，B1图中的小范围内包括B2图中的所有邻居节点。在B2中，图中的邻域包含图中的所有相邻节点，潜在空间中的邻域包含虚线圆中半径为ρ的节点。C在结构邻域上的双层聚合。
虚线箭头和实线箭头分别表示低级和高级聚合。 蓝色和绿色箭头分别表示图形中和邻近空间上的聚集。 

这三部分具体如下：
 1. 节点嵌入(node embedding )(图中A1和A2),利用graph embedding技术将图上的节点(如节点 v)映射为隐空间一个向量表示 Zv

2.  结构领域(structural neighborhood)(图中B1和B2),B2中虚线以内的节点表示节点的真实邻居，虚线以外的节点表示节点真实邻居。不同节点相对于红色节点有9种相对位置关系 [r1,r2.....r9] ，关系映射函数为$r:(Zv,Zu) —>r  \in R$

3. 双层聚合(bi-level aggregation)(图中C),聚合节点v在某个关系r下的邻居的信息.这里用一个虚拟节点的概念来表示.

####  node embedding 
-----
令G =（V，E）为图，每个节点v∈V具有特征向量xv,每个边缘e∈E连接两个节点。

记f: v -> Zv 为一个映射函数，将Graph中的节点V映射到latent space，d是隐空间的维度。在mapping的过程中，Graph的结构和属性都保留下来了，然后表现成latent space里的geometry 表示。

####    Structural neighborhood
------
基于Graph和latent space，构造了一个结构化的邻居，$N(v)=({N_g(v),N_s(v),r})$,  可以看出结构化邻居包括节点集合${N_g(v),N_s(v)}$以及边的关系r。$N_g(v)=\{  u|u \in V,(u,v) \in E \}$是图上与节点v直接连接的节点。$N_s(v)=\{  u|u \in V,d(z_u,z_v) < \rho \}$是latent space上的邻居节点,其中邻居节点到中心节点小于给定距离$\rho$。其中d(,)就是计算在隐空间中向量的距离，Ns(v)中可能包括Graph中离节点v比较远的节点同时又具有一定的相似性，保证了在隐空间中的距离比较近。通过在latent space上的邻域可以捕捉长范围的依赖关系。

关系操作r 就是输入节点v/u的有序的位置pair(Zv,Zu)，输出一个离散的变量r表示空间中从节点v到节点u的集合关系，
$$r:(z_v,z_u) \rightarrow r \in R$$

其中 R是几何关系的集合,当使用操作 r的时候，可以指定r为潜在空间中感兴趣的任意几何关系。

![image.png](https://upload-images.jianshu.io/upload_images/3426235-a66605acdba97201.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

####    Bi-level aggregation 
------
由于定义了结构化邻居N(v),因此针对于节点的更新进行了新的定义。 Bi-level aggregation 包括两个聚合函数low-level和high-level。
![](https://upload-images.jianshu.io/upload_images/3426235-418da3cfd2d2e540.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**low-level**通过函数p将garph space和latent space中类似的节点聚合在一起，虚拟节点的特征为 $e^{v, l+1}_{i,r}$,low level聚合如图C所示。
![](https://upload-images.jianshu.io/upload_images/3426235-119a48d6037061f1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
【蓝色和绿色的线代表在Graph/latent 两个space的操作】

在high-level，虚拟节点的特征 $e^{v, l+1}_{i,r}$通过函数q进行信息传递，q将虚拟节点(i,r)作为输入来区分不不同节点的不同邻居，我们可以将q设置成任何一个保序函数。根据获得的$m_v^{l+1}$加上非线性变换得到输出$h_v^{l+1}$，其中权重矩阵WL是所有节点共享的权重矩阵，$\sigma(.)$是一个非线性激活函数譬如ReLU。

####  Compare to related work
---
本文的解决办法是，通过映射到 latent 空间来解决捕捉邻居节点的问题，通过bi-level aggregation来传递信息。
对于第二个缺点，用了两个方法，
1）特征相似但是相距很远的节点可以在latent映射成临近节点，从而解决了长距离信息传递的问题，不过这对embedding方法提出了较高的要求，要求这些映射方法能够区分相似节点与不同节点。

2）结构信息使得中心节点能够区分不同的邻居节点。因而在whole graph来传递临近节点的信息。

####  Geom-GCN：An Implementation of the scheme
-----
上一节中很抽象的Low-level aggregation  p和High-level aggregation   q以及关系映射函数 r。给出了具体的形式:
关系映射函数 r考虑了4种不同的位置关系.
![](https://upload-images.jianshu.io/upload_images/3426235-8cfab36fed841d16.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Low-level aggregation  p指的是GCN中的平均操作：
![](https://upload-images.jianshu.io/upload_images/3426235-d76f067de0017f78.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

High-level aggregation  q 本质就是拼接操作.
![](https://upload-images.jianshu.io/upload_images/3426235-8db843e58edfac9b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

####  How to distinguish the non-isomorphic graphs once structural neighborhood
本文argue之前的工作没能较好的对结构信息进行描述.这里给了一个case study来说明Geom-GCN的优越性.
![](https://upload-images.jianshu.io/upload_images/3426235-cb4cd3b5e80d0b3b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
假设所有节点的特征都是a .针对节点 V1 来说,其邻居分别为 {V2,V3}和 {V2,V3,V4} . 假设采用mean或者maximum的aggregator.
-  之前的映射函数 f ,$Agg\{ f(a),f(a)\} =Agg\{ f(a),f(a),f(a)\} $ .则两种结构无法区分.

-  本文的映射函数  f i,$Agg\{ f_2(a),f_8(a)\} =Agg\{ f_2(a),f_7(a),f_9(a)\} $ .则两种结构可以区分.

###  Experiments
本文主要对比了GCN和GAT
![](https://upload-images.jianshu.io/upload_images/3426235-b2837de401e333fd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

不同数据集的homophily可以用下式衡量.
![](https://upload-images.jianshu.io/upload_images/3426235-3ae7aa505faa393f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

实验结果见下表:
![](https://upload-images.jianshu.io/upload_images/3426235-c00b6c3ac9057305.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中Isomap (Geom-GCN-I),Poincare embedding (Geom-GCN-P)，struc2vec (GeomGCN-S).采用了不同的graph embedding

作者又进一步测试了两个变种:

- 只用原始图上邻居,加上后缀-g. 如Geom-GCN-I-g
-  只用隐空间邻居,加上后缀-s. 如Geom-GCN-I-s

结果见下图:
![image.png](https://upload-images.jianshu.io/upload_images/3426235-c61a7ecea36d1285.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看出:隐空间邻居对 $\beta$较小的图贡献更大.

作者测试了不同embedding方法在选取邻居上对实验结果的影响.
![](https://upload-images.jianshu.io/upload_images/3426235-e4c7cab55a8af3ee.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看出:这里并没有一个通用的较好embedding方法.需要根据数据集来设置,如何自动的找到最合适的embedding方法是一个future work.

###  Conclusion

本文针对MPNNs的两个基础性缺陷设计了Geom-GCN来更好的捕获结构信息和长距离依赖.实验结果验证了Geom-GCN的有效性.但是本文并不是一个end-to-end的框架.有很多地方需要手动选择设计.

#### 参考
---
-  [GEOM-GCN](https://openreview.net/pdf?id=S1e2agrFvS)

- [图神经网络最新进展 ICLR2020 Geom-GCN](https://zhuanlan.zhihu.com/p/102257520)

- 19ICLR GIN How Powerful are Graph Neural Networks




















