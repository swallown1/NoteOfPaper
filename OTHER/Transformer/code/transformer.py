import torch
import torch.nn as nn

from layers import Embedding,Posting_embedding,Multihead_attention

device = torch.device('cuda')

class Encoder_Layer(nn.Module):
    def __init__(self,num_heads,emb_size):
        super(Encoder_Layer, self).__init__()
        self.emb_size = emb_size
        self.hidden=[4*emb_size,emb_size]
        self.num_heads = num_heads
        self.activation =  torch.nn.ReLU()
        self.liner1 = nn.Linear(self.emb_size, self.hidden[0], bias=True).to(device)
        self.liner2 = nn.Linear(self.hidden[0], self.hidden[1], bias=True).to(device)

        self.attention=Multihead_attention(emb_size, num_heads, causality = False).to(device)


    def feedFworad(self,x):
        outputs = self.activation(self.liner1(x))
        outputs = self.liner2(outputs)
        #残差连接
        outputs +=x
        outputs = torch.normal(outputs)
        return outputs

    def forward(self,inputs1,inputs2):
        # self-attention部分
        outputs = self.attention(inputs1,inputs2)

        #feed Forward部分
        return self.feedFworad(outputs)


class Encoder(nn.Module):
    def __init__(self,num_de,num_encoders,emb_size,num_heads=8):
        super(Encoder, self).__init__()
        self.emb_size = emb_size
        self.num_de = num_de
        self.num_encoders = num_encoders

        self.num_heads = num_heads
        # self.de_embedding = Embedding(num_en,emb_size,zero_pad=True,scale=True)
        self.en_embedding = Embedding(num_de,emb_size,zero_pad=True,scale=True).to(device)
        self.en_positional = Posting_embedding(emb_size,zero_pad=False,scale=False).to(device)

        for h in range(self.num_encoders):
            setattr(self,"encoder_"+str(h),Encoder_Layer(num_heads, emb_size).to(device))


    def forward(self,x):
        # Embedding部分
        self.enc = self.en_embedding(x)
        self.enc += self.en_positional(x)

        # 多个Encoder——layer部分
        for i in range(self.num_encoders):
            self.enc = getattr(self,"encoder_"+str(i))(self.enc,self.enc)

        return self.enc

class Decoder_Layer(nn.Module):
    def __init__(self,num_heads,emb_size):
        super(Decoder_Layer, self).__init__()
        self.emb_size = emb_size
        self.hidden=[4*emb_size,emb_size]
        self.num_heads = num_heads
        self.attention =  torch.nn.ReLU()
        self.liner1 = nn.Linear(self.emb_size, self.hidden[0], bias=True)
        self.liner2 = nn.Linear(self.hidden[0], self.hidden[1], bias=True)
        #self-attention
        self.attention1=Multihead_attention(emb_size, num_heads, causality = True).to(device)
        #encoder-decoder attention
        self.attention2=Multihead_attention(emb_size, num_heads, causality = False).to(device)


    def feedFworad(self,x):
        outputs = self.activation(self.liner1(x))
        outputs = self.liner2(outputs)
        #残差连接
        outputs +=x
        outputs = torch.normal(outputs)
        return outputs

    def forward(self,dec,enc):
        # self-attention部分
        outputs = self.attention1(dec,dec)
        # encoder-decoder attention
        outputs = self.attention2(outputs,enc)
        #feed Forward部分
        return self.feedFworad(outputs)

class Decoder(nn.Module):
    def __init__(self,num_en,num_decoders,emb_size,num_heads=8):
        super(Decoder, self).__init__()
        self.emb_size = emb_size
        self.num_en = num_en
        self.num_decoders = num_decoders

        self.num_heads = num_heads
        # self.de_embedding = Embedding(num_en,emb_size,zero_pad=True,scale=True)
        self.dec_embedding = Embedding(num_en,emb_size,zero_pad=True,scale=True).to(device)
        self.dec_positional = Posting_embedding(emb_size,zero_pad=False,scale=False).to(device)


        for h in range(self.num_decoders):
            setattr(self,"encoder_"+str(h),Encoder_Layer(num_heads, emb_size).to(device))

    def forward(self,x,enc):
        # Embedding部分
        self.dec = self.dec_embedding(x)
        self.dec += self.dec_positional(x)

        for h in range(self.num_decoders):
            self.dec = getattr(self, "encoder_" + str(h))(self.dec,enc)

        return self.dec

class Transformer(nn.Module):
    def __init__(self,num_en,num_de,emb_size):
        super(Transformer, self).__init__()
        self.num_de = num_de
        self.num_en = num_en
        self.encoder = Encoder(num_de,num_encoders=6,emb_size=emb_size).to(device)
        self.decoder = Decoder(num_en,num_decoders=6,emb_size=emb_size).to(device)

        self.logits = nn.Linear(emb_size,num_en)

    def forward(self,x,dec_inpt):

        enc = self.encoder(x)
        dec = self.decoder(dec_inpt,enc)
        logits = self.logits(dec)

        preds = torch.argmax(logits,-1).int()
        return logits,preds





