# Pytorch自带的Transformer学习
import torch
from torch import nn

#常用参数
# d_model – 编码器 / 解码器输入中预期词向量的大小(默认值=512).
# nhead – 多头注意力模型中的头数(默认为8).
# num_encoder_layers – 编码器中子编码器层(transformer
# layers)的数量(默认为6).
# num_decoder_layers – 解码器中子解码器层的数量（默认为6).
# dim_feedforward – 前馈网络模型的尺寸（默认值 = 2048).
# dropout – dropout的比例(默认值=0.1).
# activation – 编码器 / 解码器中间层，激活函数relu或gelu(默认=relu).


def test_trans():
    model = nn.Transformer(nhead=4,num_decoder_layers=6,num_encoder_layers=6,d_model=128,dim_feedforward=1024)
    ## 初始数据
    src = torch.rand(64,10,128)
    tgt = torch.rand(32,10,128)
    out = model(src,tgt)
    return out

def test_encoder():
    # torch.nn.TransformerEncoder
    # 参数：
    # coder_layer – TransformerEncoderLayer（）的实例（必需）.
    # num_layers –编码器中的子编码器(transformer layers)层数（必需）.
    # norm–图层归一化组件（可选）
    encoder_layer = nn.TransformerEncoderLayer(nhead=8,d_model=512)
    encoder = nn.TransformerEncoder(num_layers=6,encoder_layer=encoder_layer)
    src= torch.rand(10,32,512)
    out = encoder(src)
    return out


def test_decoder():
    # torch.nn.TransformerDecoder
    # coder_layer – TransformerEncoderLayer（）的实例（必需).
    # num_layers –编码器中的子编码器(transformer layers)层数（必需).
    # norm–图层归一化组件（可选）.
    decoder_layer = nn.TransformerDecoderLayer(nhead=8,d_model=512)
    decoder = nn.TransformerDecoder(decoder_layer,num_layers=6)
    memory=torch.rand(10,32,512)
    tgt=torch.rand(20,32,512)
    return decoder(tgt,memory)

if __name__ == '__main__':
    out = test_trans()
    encoder_out =test_encoder()
    decoder_out = test_decoder()
    print(decoder_out.shape)