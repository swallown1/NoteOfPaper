## ReView
----
一些自己的看法，日后看起来更方便吧！

### Screening

** 1.Abstract**

协同过滤最主要的思想就是用用更好的矢量来表示用户和商品以及来模拟用户和商品的交互过程。

** 2.Conclusion**

作者通过利用多个图形来明确表示用户-项目、用户-用户和项目-项目之间的关系。Multi-GCCF构建了三个从不同角度了解到的对可用数据的嵌入。


** 3.Q&A**
	
	为什么Multi Graph Encoding 层 只用余弦相似度找相似者，用的是1-hop之后的嵌入向量求余弦值。至于具体怎么做没讲清楚？
	
	嵌入了这么因素在里面，是否学习到的向量能很好的包含这些因素？

### literature


** 1.Data**
	Gowalla, Amazon-Books, Amazon-CDs and Yelp2018 
	然后构建用户-项目、用户-用户和项目-项目 三个图结构

** 2.Evaluation**
	采用BPR损失，以及l2正则化进行防止过拟合。
	
	评价指标：Recall@k and NDCG@k (we report Recall@20 and NDCG@20).K代表k-hop

** 3.Model**
	该模型从三个图结构中，通过GCN对可用数据的嵌入。
	
	Bipar-GCN：通过GCN分别对 user item节点进行编码嵌入向量，文中用2-hopGCN进行编码
	
	Multi Graph Encoding： 利用上一部分得到的1-hop 嵌入向量，计算user  item之间的相似度，通过sum操作聚合
	user的邻居user节点。
	
	Skip-connection：将原始的属性进行向量化，再通过一层全连接将原始属性加入进行考虑
	
	将这三个部分的结果进行整合(文中提到三种整合方式)，得到user  item的嵌入向量表示。
	
** 4.Step_conclusion**
	Bipar-GCN：得到user item 节点的 k-hop的图卷积之后的Embedding 向量，其中聚合了k-hop
	节点的信息，整合了周围节点的信息于自身。
	
	Multi Graph Encoding：聚合user(item)之间的本身含有的联系。
	
	Skip-connection：充分考虑各节点自身的属性，将节点原始的嵌入向量表示在此加入到下一步模型。
	

### Intensive_reading

** 1.Process data**
	处理数据，通过原始数据构建上述那三中图建构

** 2.Detail of model**
	
	Bipar-GCN： 通过矩阵处理，可以处理一些长尾数据。这里进行了2-hop，但是输出的结果是将1-hop和2-hop的结果拼接操作。在学习2-hop时 在1-hop的结果上加入原始eu
		
	Multi Graph Encoding：将每个user(item)的邻居节点进行sum aggregator操作，这里没有进行GCN的聚集，而是简单的相邻user(item)向量的相加，再激活。

	
	Skip-connection：将原始向量eu(ev)的结果 通过一个一层全连接，再到Information Fusion进行拼接处理。
	
	
	

### Reflect

** 参考**

通过多幅图的学习嵌入来尽可能使得最后的嵌入向量表示更具有代表性。

在进行多次处理后，尽量在加上之前的原始向量，保证原始特征不会消失，或者原始特征信息的丢失。

将GCN卷积过的Embedding 放入 用户用户  项目项目图中，可以更具有表达效果

 ** Improvement point ** 

Multi Graph Encoding处理有些简单，我认为可以做一些GCN卷积操作可能效果更好一些。
