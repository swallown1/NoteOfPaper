# RCF论文个人解析
## 摘要
该论文是对传统ICF（基于物品的协同过滤）的加强，传统的ICF主要受协同相似性关系的影响：用户的交易行为中的评级和购买。然而在真实场景中存在多重关系：两部电影有相同的导演或者两个产品相互补充（羽毛球与羽毛球拍）。不同于用户角度交互模式的协同相似性，这些物品之间的关系从元数据和功能方面揭示了细粒度知识。（怎么理解 meta-data 和 functionality?）

### 提出问题：
如何整合多元数据关系？（现有推荐系统很少涉及）

### 解决问题：
1、关系类型：同一导演

2、关系值：导演名

3、模型：用两级注意力机制

<1> 第一层：确定哪些关系类型重要？

<2> 第二层：通过关系类型中的关系值衡量历史物品的重要性？？？

通过关系值的权重确定历史物品的贡献还是通过历史物品确定哪些关系值重要？(在模型中)
