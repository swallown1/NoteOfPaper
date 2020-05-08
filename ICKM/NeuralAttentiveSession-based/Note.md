论文：Neural Attentive Session-based Recommendation

作者： Jing Li，Pengjie Ren，Zhumin Chen

来源： ICKM 17

代码：https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch/

----

#### Session-based
基于会话的推荐是通过利用客户捕获用户在使用时留下的一些小的行为信息。但是基于会话的推荐方式是无法获取用户的基本信息的，因此传统的MF，CF是无法使用的。所以在基于会话的推荐中，所有的信息就是用户点击的item的信息，所以绝大多数模型普遍都是item-item模型，只能考虑商品本身的属性。同时由于点击商品是一个序列，因此其中还包含着时间的信息，因此绝大多数的模型都会通过RNN等来提取时间序列的信息，这篇论文也是在GRU4rec的基础上进行改进。

####  论文的写作动机
因此我们的问题是对序列信息进行建模，传统的方法并不能建模session中item的连续偏好信息。

**传统方法的问题：**
Markov decision Processes ：
马尔科夫决策过程，用四元组<S,A, P, R>（S: 状态, A: 动作, P: 转移概率, R: 奖励函数）刻画序列信息，通过状态转移概率的计算点击下一个动作：即点击item的概率。缺点：状态的数量巨大，会随问题维度指数增加。[增强学习（二）----- 马尔可夫决策过程MDP](https://link.zhihu.com/?target=http%3A//www.cnblogs.com/jinxulin/p/3517377.html)）

**深度学习方法**：
深度学习的方法，从RNN，GRU到后面更复杂的结构，有很多，是一个系列的问题。[推荐中的序列化建模：Session-based neural recommendation](https://zhuanlan.zhihu.com/p/30720579)

####  NARM 细节
1. session-based Recommendation
 Session-based  recommendation 是在给定用户当前浏览item序列来预测用户下一个点击的item，那么是如何定义这个问题呢？

[x1, x2, ..., xn−1, xn]，表示点击的session，其中xi指的是点击一次item的索引。
对于模型M，在给定点击序列X== [x1, x2, ..., xt−1, xt], 1 ≤ t ≤ n后，模型的输出为
y = [y1,y2, ...,ym−1,ym]，这里的m指的是所有的item，yi指的是第i个item的点击的可能性。最后在y中推荐top-K个item作为结果。

2. Overview
本文提出的NARM是一个基于encoder-decoder结构的方法。
最基本的想法是对当前的session构建一个隐层表示，通过其来生成预。

通过下面的图，可以看出encoder将X [x1, x2, ..., xt−1, xt],转化成一系列高维的隐层表示h = [h1, h2, ..., ht−1, ht] 
![image.png](https://upload-images.jianshu.io/upload_images/3426235-78fcd8103d6bcf9b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 αt指的是t时刻的注意力权值，将其送入session feature generator来构建当前session的表示，来对t时刻进行decoder，得到ct. ct在通过矩阵U进行转换，通过激活函数得到排序序列y = [y1,y2, ...,ym−1,ym]。

αt是用来决定隐层状态的注意力权值。αt是可以动态设定的，αt可以是隐层状态表示函数，或者是item嵌入表示。

 global encoder用来建模用户序列行为，local encoder用来捕获用户的主要目的。

3.  Global Encoder in NARM
 Global Encoder 用来建模用户的序列行为，输入是之前的点击序列，输出是用户序列行为的特征。输入和输出都是正则化的高维向量。

![](https://upload-images.jianshu.io/upload_images/3426235-84cfb741964c0620.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如图 是global encoder的结构图，这里使用的是GRU作为循环单元，因为GRU可以消除RNN的梯度消失问题。

说说GRU的细节吧

GRU的激活是先前激活ht - 1和候选激活Dht之间的线性插值：
![](https://upload-images.jianshu.io/upload_images/3426235-b59fbc41207142c5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中Zt是更新门，
![](https://upload-images.jianshu.io/upload_images/3426235-0efa8b0c57c0b2d9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中xt指t时刻的输入，ht-1指的是GRU的前一个隐层状态，通过这两部分探究xt和前部分的关联程度，Zt表示更新的程度。

![](https://upload-images.jianshu.io/upload_images/3426235-cf17f33c6abfcf05.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中rt为重置门，其定义为
![](https://upload-images.jianshu.io/upload_images/3426235-42e3c988c8429bb5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

使用简单的会话特性生成器，我们基本上使用最终的隐藏状态ht作为用户连续行为的表示
$$c^g_t= ht$$
然而，这种全局编码器也有其缺点，例如对整个序列行为的矢量化概括通常很难捕获当前用户的精确意图。因此设计了Local Encoder。

4.  Local Encoder in NARM
先看一下其总体的结构
![](https://upload-images.jianshu.io/upload_images/3426235-25c927a691c1be40.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Local Encoder 的结构和 Global Encoder的结构类似，使用GRU作为基础的组成部分。为了捕获序列session中的主要目的，设计了item级别的注意力机制，动态选择和线性组合输入序列的不同部分：
![](https://upload-images.jianshu.io/upload_images/3426235-bf1152a43c2ac857.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中加权因子α确定哪些输入序列的一部分应该强调或忽略了在进行预测时,进而是隐状态的函数,
![](https://upload-images.jianshu.io/upload_images/3426235-05ef9a8a43c01194.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

权重因子αt j用来平衡输入位置j和输出位置t之间的关联程度，因此可以视为匹配模型。在Local Encoder 中  q函数是计算最后隐层ht和之前点击item的向量表示hj的相似度。
![](https://upload-images.jianshu.io/upload_images/3426235-8a59fe2e9a7313d3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中矩阵A1用来将ht转化到一个潜在空间，A2作用在hj。local encoder的好处是自适应的专注于session中对于捕获用户主要兴趣的一些重要的item。


5.  NARM Model
![image.png](https://upload-images.jianshu.io/upload_images/3426235-e72621634647f05b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从前面的介绍可以看出 global encoder是对于整个序列行为的汇总， local encoder则是用来捕获session序列中对于捕获用户兴趣的比较重要的item。我们推测，序列行为的表示可能为捕获当前会话中用户的主要目的提供有用的信息。因此，我们使用顺序行为的表示形式和先前的隐藏状态来计算每个单击项的注意权重。

下图中，hgt是global encoder的输出，合并到ct中，为NARM提供一个顺序行为表示。应该注意的是，NARM中的会话特征生成器将在全局编码器和本地编码器中调用不同的编码机制，尽管它们稍后将组合起来形成统一的表示。同时hgt相对于
 local encoder每个hlt有很大的不同，前者负责对整个顺序行为进行编码。后者用于计算前一隐藏状态下的注意权值。这样，将这两部分进行结合，使得用户的顺序行为和当前会话中的主要目的都可以建模为统一的表示ct。而ct是由cgt和clt拼接而成：
![](https://upload-images.jianshu.io/upload_images/3426235-7086f5f76db9b9dc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上图还给出了NARM中采用的解码机制的图形说明。通常，一个标准的RNN使用全连接层来解码，但是使用全连接层意味着，那么每层的学习参数为|H| ∗ |N |，其中|H|是session表示的维度，|N|是预测item的数，需要大量的存储空间。尽管可以使用hierarchical softmax layer和负采样的方法可以减少参数，但是这对于此模型不是最好的选择。
本文提出了一种新的双线性译码方案( bi-linear decoding scheme)，既减少了参数的数目，又提高了NARM的性能。具体来说，当前会话的表示形式和每个候选项之间的双线性相似函数用于计算相似度评分Si。*bi-linear的理解就是将后选的item和ct的嵌入表示计算两种的相似程度*
![](https://upload-images.jianshu.io/upload_images/3426235-ff8457052d3c6903.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中B( |D| ∗ |H|)就是为了将ct的嵌入表示转化成和Embedding相同的维度(|D|) ，然后，将每个项目的相似度评分输入到softmax层，以获得该项目接下来出现的概率。使用bi-linear decoder 可以将参数的数量由|N | ∗ |H| 变成 |D| ∗ |H| ，而且结果还很好。
![](https://upload-images.jianshu.io/upload_images/3426235-d09a9c006da4ebac.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


本文在学习模型参数的时候没有并行处理每个session，而是分别处理每个session序列e[x1, x2, ..., xt−1, xt]，训练时采用小批量正则化梯度下降法，同时使用交叉熵损失
![](https://upload-images.jianshu.io/upload_images/3426235-8c96eb1d374924f5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
q是预测值，p是真实标签。

### 实验
1.数据集
YOOCHOOSE：此数据集包含电子商务网站的点击流。在过滤掉长度为1的会话和出现次数少于5次的条目之后，还剩下7981580个会话和37483个条目。

DIGINETICA：DIGINETICA2来自2016 CIKM杯。我们只使用发布的事务数据，还过滤了长度为1的会话和出现次数少于5次的条目。最后，数据集包含204771个会话和43097个条目
![image.png](https://upload-images.jianshu.io/upload_images/3426235-e6335c34654c02fa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们首先对两个数据集进行了一些预处理。对于YOOCHOOSE，我们使用随后一天的会话进行测试，并从测试集中过滤掉没有在训练集中出现的单击项。对于DIGINETICA，惟一的区别是我们使用随后一周的会话进行测试。由于我们没有以会话并行方式来训练NARM，因此需要对序列分割预处理。对于输入会话[x1, x2，… xn−1,xn]，我们生成序列和相应的标签([x1]，V (x2)， ([x1, x2]，V (x3)，…， ([x1, x2，… xn−1]，V (xn))，用于YOOCHOOSE和DIGINETICA的培训。对应的标签V (xi)是当前会话中的最后一次单击。

2. 其他模型对比
NARM于5个传统方法(POP, S-POP, Item-KNN, BPR-MF and FPMC)以及两个以RNN为基础的模型(GRU-Rec and Improved GRU-Rec)
![image.png](https://upload-images.jianshu.io/upload_images/3426235-e713bc2d806bea4c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

3.实验评估
由于推荐系统每次只能推荐几个项目，用户可能选择的实际项目应该在列表的前几个项目中。因此，我们使用以下指标来评估推荐列表的质量。
•Recall@20:主要的评估指标是Recall@20，这是所有测试用例中所需项目位于前20项中的情况比例。Recall@N不考虑项目的实际排名，只要它在前n位，并且通常与其他指标(如点击率(CTR)[21])密切相关。
•MRR@20:另一个使用的指标是MRR@20(平均倒数排名)，这是愿望物品倒数排名的平均值。如果秩大于20，则倒数秩设置为0。MRR考虑了项目的级别，这在推荐顺序很重要的情况下是很重要的。
 
4.分析部分
本文从模型的组成部分和序列长度进行了分析。
对于模型的组成，当然是同时考虑global encoder和local encoder的时候是最好的。
对于序列的长度，在4-17个的时候最佳。会话长度特长的时候，效果会下降，本文认为原因是，当一个会话太长时，用户很可能会盲目地点击一些条目，这样NARM中的本地编码器就无法捕获当前会话中用户的主要目的。


### 代码部分
先给出该模型的一个pytorch版本的[代码地址]()
主要是看看模型实现的细节部分

1.  init参数部分

 self.emb将所有的item转化成 self.embedding_dim维度的嵌入向量。

self.gru 是模型中的GRU基础单元， 其中nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers) ，self.n_layers表示GRU的个数，self.embedding_dim表示输入的xt的维度，self.hidden_size表示隐层输出(ht)  的维度。
```
def __init__(self, n_items, hidden_size, embedding_dim, batch_size, n_layers = 1):
        super(NARM, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx = 0)
        self.emb_dropout = nn.Dropout(0.25)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        #self.sf = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

2. 模型的细节

embs ：将序列中的每个X转化成embedding，

self.b:是公式10中的B，目的是将Ct 从2*self.embedding_dim 转化成 self.embedding_dim

```

    def forward(self, seq, lengths):
          # gru中  隐层的矩阵参数
        hidden = self.init_hidden(seq.size(1))
        embs = self.emb_dropout(self.emb(seq))
        # 将embs 按照lengths进行压缩
        embs = pack_padded_sequence(embs, lengths)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)

        # 去最后一个GRU的隐层作为 hgt
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)
          #  hgt作为文中的 c_global 
        c_global = ht

        #每一个GRU的输出，即hj，将其进行线性转换   Whj
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())  
        # 将ht进行线性转换  Whi
        q2 = self.a_2(ht)
        
        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device = self.device), torch.tensor([0.], device = self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand
        # 对应文章公式8  计算att 注意力权值
        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())

        #计算文中的local encoder   α*(Whj+Wht)
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)
        
        #计算得到文中的Ct
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        
        #将所有的item进行Embedding
        item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        #公式10
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))
        #经过softmax
        # scores = self.sf(scores)

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)
        
```



