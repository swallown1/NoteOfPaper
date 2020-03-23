## Deep Neural Networks for YouTube Recommendations

### ABSTRACT

YouTube是现有规模最大，最复杂的行业推荐系统之一。 在本文中，我们从较高的角度描述该系统，并着重于深度学习带来的显着性能提升。 本文根据经典的两阶段信息检索二分法进行了拆分：首先，我们详细介绍了一个深层候选生成模型，然后描述了一个单独的深度分类模型。 我们还将提供从设计，迭代和维护庞大的推荐系统中获得的实践经验和见解，这些推荐系统会对用户产生巨大的影响。

###  INTRODUCTION

在本文中，我们将重点介绍深度学习最近对YouTube视频推荐系统产生的巨大影响。

从三个主要方面来看，推荐YouTube视频具有极大的挑战性：

1. 规模：许多现有的推荐算法被证明可以很好地解决小问题，但无法达到我们的规模。

2.  新鲜度：YouTube具有非常动态的语料库，每秒上传许多小时的视频。 推荐系统应具有足够的响应能力，以对新上传的内容以及用户采取的最新操作进行建模。

3. 噪音：噪音主要体现在用户的历史行为往往是稀疏的并且是不完整的，并且没有一个明确的ground truth的满意度signal，我们面对的都是noisy implicit feedback signals。噪音另一个方面就是视频本身很多数据都是非结构化的。这两点对算法的鲁棒性提出了很高的挑战。

之所以要在推荐系统中应用DNN解决问题，一个重要原因是google内部在机器学习问题上的通用solution的趋势正转移到Deep learning，系统实际部署在基于tensorflow的Google Brain上。

### SYSTEM OVERVIEW

首先看一下系统的整体结构

![ overall structure](https://github.com/swallown1/NoteOfPaper/blob/master/RecSys/DNNForYouTube/images/voerview.png)

在工业中，整个推荐系统部分分为 candidate generation 和Ranking两部分。

对于candidate generation部分主要是从用户的YouTube活动历史记录中获取事件作为输入，并从大型语料库中检索一小部分（数百个）视频。这些候选视频旨在与用户高度相关，而网络的生成只通过协同过滤提供广泛的个性化。(其中用户之间的相似性是根据诸如观看过视频的ID，搜索查询指令和受众特征之类的粗略特征来表达的。)

Ranking 通过使用描述视频和用户的丰富功能集，根据所需的目标功能为每个视频分配分数，从而完成了此任务。  Ť

之所以把推荐系统划分成Candidate geeration 和Ranking两个阶段，主要是从性能方面考虑的。Candidate geeration 阶段面临的是百万级视频，单个视频的性能开销必须很小；而Ranking阶段的算法则非常消耗资源，不可能对所有视频都算一遍，实际上即便资源充足也完全没有必要，因为往往来说通不过Candidate geeration 粗选的视频，大部分在Ranking阶段排名也很低。接下来分别从Candidate geeration 和Ranking阶段展开介绍。

### CANDIDATE GENERATION

我们的神经网络模型的早期迭代通过浅层网络模拟了这种分解行为，该浅层网络仅嵌入了用户以前的观看记录。

#### 推荐分类

本文提出的推荐是一个大规模的分类问题。即在时刻t，为用户U（上下文信息C）在视频库V中精准的预测出视频i的类别（每个具体的视频视为一个类别，i即为一个类别），用数学公式表达如下：

![](https://github.com/swallown1/NoteOfPaper/blob/master/RecSys/DNNForYouTube/images/math1.png)

上面的式子可以看出是一个SoftMax多分类器。向量u是<user, context>的一个Embedding表示，向量Vj则是视频j的一个Embedding向量表示。所以DNN的目标就是在用户信息和上下文信息为输入条件下学习用户的embedding向量u。用公式表达DNN就是在拟合函数

$$f_{DNN}(user_{info},content_{info})$$

而这种超大规模分类问题上，至少要有几百万个类别，实际训练采用的是Negative Sampe，类似于word2vec的Skip-Gram方法



#### Model Architecture

![模型](https://github.com/swallown1/NoteOfPaper/blob/master/RecSys/DNNForYouTube/images/model.png)

整个模型是有三个隐层的DNN结构。该模型的输入是由用户的观看历史，收索历史，人口统计信息和上下文信息concat 成定长的输入向量；输出分线上和离线训练两个部分。

离线训练阶段输出层为softmax层，输出2.1公式表达的概率。而线上则直接利用user向量查询相关商品，最重要问题是在性能。我们利用类似局部敏感哈希


#### Heterogeneous Signals

类似于word2vec的做法，每个视频都会被embedding到固定维度的向量中。用户的观看视频历史则是通过变长的视频序列表达，最终通过加权平均（可根据重要性和时间进行加权）得到固定维度的watch vector作为DNN的输入。

**使用深度神经网络作为矩阵分解的一般化方法的主要优势在于，可以轻松地将任意连续和分类特征添加到模型中。**

搜索历史记录与观看历史记录的处理方式相似-每个查询都被标记为unigram和bigrams，并且每个标记都被嵌入。 取平均后，用户的标记化嵌入查询代表了汇总的密集搜索历史记录。

人口统计特征对于提供先验条件很重要，因此建议对于新用户而言行为合理。

用户的地理区域和设备已嵌入并连接在一起。 简单的二进制和连续功能（例如用户的性别，登录状态和年龄）直接以标准化为[0，1]的实际值直接输入到网络中。

主要特征：

历史搜索query：把历史搜索的query分词后的token的embedding向量进行加权平均，能够反映用户的整体搜索历史状态
人口统计学信息：性别、年龄、地域等
其他上下文信息：设备、登录状态等

#### “Example Age” （视频上传时间）特征

视频网络的时效性是很重要的，每秒YouTube上都有大量新视频被上传，而对用户来讲，哪怕牺牲相关性代价，用户还是更倾向于更新的视频。当然我们不会单纯的因为一个视频新就直接推荐给用户。

因为机器学习系统在训练阶段都是利用过去的行为预估未来，因此通常对过去的行为有个隐式的bias。视频网站视频的分布是高度非静态（non-stationary）的，但我们的推荐系统产生的视频集合在视频的分布，基本上反映的是训练所取时间段的平均的观看喜好的视频。因此我们我们把样本的 “age” 作为一个feature加入模型训练中。从下图可以很清楚的看出，加入“example age” feature后和经验分布更为match。

![图4](https://github.com/swallown1/NoteOfPaper/blob/master/RecSys/DNNForYouTube/images/f4.png)


####  Label and Context Selection

推荐问题经常会涉及一些替代的问题，或者是将问题转化成特定的上下文，例如准确预测收视率可以更好的进行进行电影推荐。替代学习问题对于线上A/B测试来说很重要但是对于线下实验来说比较困难。

