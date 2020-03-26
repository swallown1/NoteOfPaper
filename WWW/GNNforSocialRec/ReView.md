## ReView

### Screening

** 1.Abstract**

社会推荐存在的问题是：社交关系中，用户涉及两个图。一个是用户项目图，一个是用户-用户社交图。**如何将这两个图进行联合，以及如果通过学习者两个图，将用户和项目更好的用向量表示？**

论文要解决的问题：

 1. 第一个挑战是如何固有地组合这两个图。 此外，用户项目图不仅包含用户与项目之间的交互，还包括用户对项目的意见。 

 2. 第二个挑战是如何共同捕获用户与项目之间的交互和观点。

 3.第三个挑战是如何区分具有不同优势的社会关系。

** 2.Conclusion**

作者通过结合了GCN的优势（用于对社交网络中的扩散过程进行建模）和经典的基于潜在因子的模型来捕获用户项目偏好。

模型的输入：

	1. 模型的输入其实就是两个图(user-item,user-user)
	
	2. 将user item rate 进行向量化，进行Aggre,通过user-item学习含有隐藏item兴趣的user嵌入向量。再通过嵌入向量放到 user-user图中进行学习，得到邻居节点的嵌入向量。
	
	3. 将 item 通过user-item学习含有隐藏user兴趣的item嵌入向量。

	4.通过上面的 user 和item的嵌入向量，分别将其进行标准化和drop之后，连接进行两层神经网络 最后输出得分。

模型的输出：

	模型的输出是通过上面建立好的两个嵌入字典，对测试节点进行嵌入，再通过两层神经进行模拟交互，得到最终的得分情况。

** 3.Q&A**

	为什么使用relu函数？

	其次在神经网络中没加入偏值，同时在网络中，每一层都加入了正则化，为什么？


### literature


** 1.Data**
	两个图(user-item,user-user)，即用户对项目的打分，以及用户和用户的关联

** 2.Evaluation**
	平方损失函数，通过RMSprop作为优化器

** 3.Model**
	该模型包含三个组件：用户建模，项目建模和评级预测。

	用户建模：它是要学习用户的潜在因素，分别通过两个图(item-user,user-user)进行学习

	项目建模: 用于学习项目的潜在因素。从item-user图中学习item的潜在因素。

	评级预测：通过集成用户和项目建模组件来通过预测学习模型参数。
	
** 4.Step_conclusion**
	
	1.user Aggre ：得到的是将用户进行嵌入，得到向量表示

	2.item Aggre ： 得到的是item的一个嵌入向量表示

	3.GraphRec ：通过建模交互过程，得到预测打分情况。



### Intensive_reading

** 1.retrieve data**
	将数据进行处理，整理成 user 对 item的 评分(Rate)
	```
	# 三个元素的元祖的集合  是用户u对项目v的打分r   14091*()
	trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
	```

** 2.Process data**
	根据user item rate的个数进行初始化嵌入字典，为其进行聚合作为初始化。

	```
	u2e = nn.Embedding(num_users, embed_dim).to(device)
	v2e = nn.Embedding(num_items, embed_dim).to(device)

	# 将评分也转化成Embedding
	r2e = nn.Embedding(num_ratings, embed_dim).to(device)
	```

** 3.Detail of model**

	1. gv() 为两层神经网络，即注意力机制中a 的求法中的部分
	```
			x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))
			#求软注意力机制的 alpha
            att_w = self.att(o_history, uv_rep, num_histroy_item)
			#执行 mat1 和 mat2 的矩阵乘法.
			#gv() 为两层神经网络
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()
	```

	2. 通过user-item图聚合user的嵌入向量，在通过该嵌入向量做为输入通过user-user图聚合user的嵌入向量。

	```
	# user feature
    # features: item * rating
	agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
	enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
	    # neighobrs
	agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
	enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
	                        base_model=enc_u_history, cuda=device)

	```

### Reflect

 ** 参考**

 用户的打分，来表示用户的一个喜好程度，分数高低进行判断用户是喜欢还是不喜欢。

这篇论文的优秀点在于很好的融合了user-item 和item-item两个图的方式，使得学习到的user item的嵌入很好，所以在处理社会推荐的问题，可以参考这样的模型。

此模型的前一部分类似于一个处理，我觉得以后可以将这个模型做base_model，更好的来学习user item的嵌入向量。


 ** Improvement point ** 

 对于初始的user 和item的节点，通过的是初始化的嵌入向量，我认为这里可以用一些item自己的特性进行处理得到的嵌入向量作为聚合的初始向量 效果会更好一点。

 对于Aggre部分，采用的是图神经网络的思想，但是我觉得此部分可以更改一下，改成通过图卷积网络的方式进行信息提取会不会更好一点？

 对于交互建模部分，次模型采用的是两层神将网络进行交互建模，类似于FM中 采用的是矩阵相乘的方式模拟交互建模。所以对此我觉得可以对交互建模这部分进行改进，需要多看一点这方面的论文？