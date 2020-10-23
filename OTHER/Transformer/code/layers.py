import torch
import numpy as np
import torch.nn as nn


device = torch.device('cuda')

class Embedding(nn.Module):
    def __init__(self,num_vocab,emb_size,zero_pad=True,scale=True):
        super(Embedding, self).__init__()
        self.emb_size = emb_size
        self.zero_pad = zero_pad
        self.scale = scale

        self.Embedding = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_vocab,self.emb_size)))

        if self.zero_pad:
            self.Embedding.weight = torch.cat((torch.zeros(1,self.emb_size),self.Embedding[1:,:]),0)

    def forward(self,x):
        outputs = self.Embedding[x.long()]  # B * max_len * emb_size

        if self.scale:
            outputs = outputs * (self.emb_size**0.5)

        return outputs

class Posting_embedding(nn.Module):
    def __init__(self,emb_size,zero_pad=True,scale=True):
        super(Posting_embedding, self).__init__()
        self.emb_size = emb_size
        self.zero_pad =zero_pad
        self.scale =scale

    def forward(self,x):
        B,Max_len = list(x.shape)
        #位置索引
        position_index = torch.range(0,Max_len-1).unsqueeze(0).repeat(B,1).to(device) #B * Max_len

        position_enc = np.array([
            [pos / np.power(10000, 2.*i / self.emb_size) for i in range(self.emb_size)]
            for pos in range(Max_len)])

        position_enc[:,0::2] = np.sin(position_enc[:,0::2]) # dim 2i
        position_enc[:,1::2] = np.cos(position_enc[:,1::2]) # dim 2i+1

        lookup_table = torch.tensor(position_enc)
        lookup_table = lookup_table.to(device)
        if self.zero_pad:
            lookup_table = torch.cat((torch.zeros(1,self.emb_size),lookup_table[1:,:]),0)

        output = lookup_table[position_index.long()]

        if self.scale:
            output = output * self.emb_size **0.5

        return output



class ScaledDotProductAttention(nn.Module):
    def __init__(self,emb_size,causality=False):
        super(ScaledDotProductAttention, self).__init__()
        self.causality = causality
        self.Q=nn.Linear(emb_size,emb_size)
        self.K=nn.Linear(emb_size,emb_size)
        self.V=nn.Linear(emb_size,emb_size)

        self.activation = torch.nn.ReLU()

    def forward(self,x):
        q = self.activation(self.Q(x))
        k = self.activation(self.K(x))
        v = self.activation(self.V(x))

        outputs = torch.bmm(q,k.transpose(2,1))
        outputs = outputs / (list(k.shape)[-1]**0.5)

        # 由于 idex=0的embedding是0，因此可以对idex=0部分的设置很小的值，这样算注意力机制的时候不会产生贡献
        key_masks = torch.sin(torch.abs(torch.sum(x, -1)))
        key_masks = key_masks.unsqueeze(1).repeat(1,list(x)[1],1)
        # 去除idex=0的embedding对计算注意力带来的影响。
        # 这是对Encoder部分这么计算注意力值
        padding = torch.ones_like(outputs)*(-2**32+1)
        outputs = torch.where(torch.equal(key_masks,0),padding,outputs)

        # 对于decoder部分  self-attention 部分应该是对角线及一下部分为1，其他部分为0
        if self.causality:
            diag_vals = torch.ones_like(outputs[0, :, :])
            tril = torch.tril(diag_vals)
            masks = tril.unsqueeze(0).repeat(list(outputs)[0],1,1)

            paddings = torch.ones_like(masks) * (-2 ** 32 + 1)
            outputs = torch.where(torch.equal(masks, 0), paddings, outputs)

        outputs = torch.softmax(outputs)
        # Query Mask  这里直接设置0
        query_masks = torch.sin(torch.abs(torch.sum(x, -1)))
        query_masks = query_masks.unsqueeze(1).repeat(1,1,list(x)[1])
        outputs *=query_masks

        outputs = torch.matmul(outputs,v)
        #残差连接
        outputs += q
        #正则化
        outputs = torch.normal(outputs)
        return outputs

class Multihead_attention(nn.Module):
    def __init__(self,emb_size,num_heads,causality=False):
        super(Multihead_attention, self).__init__()
        self.num_heads=num_heads
        self.causality = causality
        self.emb_size = emb_size
        self.Q=nn.Linear(emb_size,emb_size)
        self.K=nn.Linear(emb_size,emb_size)
        self.V=nn.Linear(emb_size,emb_size)

        self.activation = torch.nn.ReLU()

    def forward(self,x1,x2):
        q = self.activation(self.Q(x1))
        k = self.activation(self.K(x2))
        v = self.activation(self.V(x2))

        # torch.split 第二个参数是指分割后的维度  不同于tf.split中参数二，其指的是分割成几份
        q_=torch.cat(torch.split(q,self.emb_size//self.num_heads,2),0)
        k_=torch.cat(torch.split(k,self.emb_size//self.num_heads,2),0)
        v_=torch.cat(torch.split(v,self.emb_size//self.num_heads,2),0)


        outputs = torch.bmm(q_,k_.transpose(2,1))
        outputs = outputs / (list(k.shape)[-1]**0.5)

        # 由于 idex=0的embedding是0，因此可以对idex=0部分的设置很小的值，这样算注意力机制的时候不会产生贡献
        key_masks = torch.sin(torch.abs(torch.sum(x2, -1)))
        key_masks = key_masks.repeat(self.num_heads,1)

        key_masks = key_masks.unsqueeze(1).repeat(1,list(x1.shape)[1],1)
        # 去除idex=0的embedding对计算注意力带来的影响。
        # 这是对Encoder部分这么计算注意力值
        padding = torch.ones_like(outputs)*(-2**32+1)
        # print(x1.shape)
        # print(key_masks.shape)
        # print(padding.shape)
        # print(outputs.shape)
        # 主要学习一下le  lt eq equal的区别
        outputs = torch.where(torch.le(key_masks,0),padding,outputs)

        # 对于decoder部分  self-attention 部分应该是对角线及一下部分为1，其他部分为0
        if self.causality:
            diag_vals = torch.ones_like(outputs[0, :, :])
            tril = torch.tril(diag_vals)
            masks = tril.unsqueeze(0).repeat(list(outputs)[0],1,1)

            paddings = torch.ones_like(masks) * (-2 ** 32 + 1)
            outputs = torch.where(torch.le(masks,0), paddings, outputs)

        outputs = torch.nn.functional.softmax(outputs)
        # Query Mask  这里直接设置0
        query_masks = torch.sin(torch.abs(torch.sum(x1, -1)))
        query_masks = query_masks.repeat(self.num_heads,1)
        query_masks = query_masks.unsqueeze(-1).repeat(1,1,list(x2.shape)[1])
        outputs *=query_masks

        outputs = torch.matmul(outputs,v_)
        # 将算出的多个子embedding拼接回去
        # aa=torch.split(outputs,list(outputs.shape)[0]//self.num_heads,0)

        outputs = torch.cat(torch.split(outputs,list(outputs.shape)[0]//self.num_heads,0),2)
        #残差连接
        outputs += q
        #正则化
        outputs = torch.normal(outputs)
        return outputs

def label_smoothing(inputs,epsilon=0.1):
    K = list(inputs.shape)[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)



