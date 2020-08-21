## SIGIR 2020 papers


### Graph-Based
1、Hierarchical Fashion Graph Network for Personalized Outfit Recommendation  
* [论文](../paper/HFGN.pdf)
* [代码](https://github.com/xcppy/hierarchical_fashion_graph_network)

![](2020_papers_files/1.jpg)

这篇论文主要的将时尚兼容性建模和个性化服装推荐。什么意思呢？就是通过不同单个商品的组合，将组合后的商品推荐给用户。于是，用户提出一种分层时装图神经网络，
主要的做法是，通过将信息从较低级别传播到更高级别来改进节点嵌入，即从项目节点收集信息以更新服装表示，然后通过历史服装来增强用户表示。如图，
他的做法是先通过单个物品的向量通过GCN聚合得到服装组合的向量表示，用户则通过历史购买过的服装组合，通过图卷积得到用户的向量表示，在给用户进行推荐。
不过这里存在很多的问题？1、用户历史记录中如果商品很少或者没有服装组合怎么办？2、如果单个商品数量多，那进行组合的可能行也多，怎么选择商品的组合，如果
全组合肯定也存在巨大的问题？






