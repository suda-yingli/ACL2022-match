import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from pytorch_pretrained_bert.modeling import BertModel

from scalarmix import ScalarMix


class Bert_Embedding(nn.Module):
    def __init__(self, bert_path, bert_layer, bert_dim, freeze=True):
        super(Bert_Embedding, self).__init__()
        self.bert_layer = bert_layer
        self.bert = BertModel.from_pretrained(bert_path)
        self.scalar_mix = ScalarMix(bert_dim, bert_layer)

        if freeze:
            self.freeze()

    def forward(self, subword_idxs, subword_masks, token_starts_masks, sens):
        sen_lens = token_starts_masks.sum(dim=1)#句子中的单词个数
        bert_outs, _ = self.bert(
            subword_idxs,
            token_type_ids=None,
            attention_mask=subword_masks,
            output_all_encoded_layers=True,
        )
        bert_outs = bert_outs[len(bert_outs) - self.bert_layer : len(bert_outs)]#获取后四层的表示，这个是一个list，里面存放了所有层的表示
        #print("embed bert outs bert_outs[0].size()",bert_outs[0].size())
        #print("embed bert outs",bert_outs.shape())
        bert_outs = self.scalar_mix(bert_outs)#输入是一个list里面有四个[3,10,768],输出是一个[3,10,768]
        #print("embed after scalar mix",bert_outs.size())
        bert_outs = torch.split(bert_outs[token_starts_masks], sen_lens.tolist())#输出是一个tuple,每个是一个句子的tensor[词数，768]
        for i in range(len(bert_outs)):
            if (bert_outs[i].size()[0] != sum(token_starts_masks[i]) and bert_outs[i].size()[0] != len(sens[i])):
                print("miss match bert with token strart")
                print("bert_outs[i]",bert_outs[i])
                print("sen[i]",sens[i])
        #print("embed after split bert outs len ",bert_outs)
        #print("embed after split bert outs ",bert_outs[0].size())#[4, 768]
        #print("embed after split bert outs ",bert_outs[1].size())#[3, 768]
        #print("embed after split bert outs ",bert_outs.size())
        bert_outs = pad_sequence(bert_outs, batch_first=True)
        #print("embed after pad bert outs ",bert_outs)
        #print("embed after pad bert outs",bert_outs.size())
        return bert_outs

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False


class Bert_Encoder(nn.Module):
    def __init__(self, bert_path, bert_dim, freeze=False):
        super(Bert_Encoder, self).__init__()
        self.bert_dim = bert_dim
        self.bert = BertModel.from_pretrained(bert_path)

        if freeze:
            self.freeze()

    def forward(self, subword_idxs, subword_masks, token_starts_masks):
        sen_lens = token_starts_masks.sum(dim=1)
        bert_outs, _ = self.bert(
            subword_idxs,
            token_type_ids=None,
            attention_mask=subword_masks,
            output_all_encoded_layers=False,
        )
        bert_outs = torch.split(bert_outs[token_starts_masks], sen_lens.tolist())
        bert_outs = pad_sequence(bert_outs, batch_first=True)
        return bert_outs

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False
