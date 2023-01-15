import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def din_padding_mask(seq):
    # 获取为 0的padding项
    seq = seq == 0
    # 扩充维度用于attention矩阵
    return seq[:, np.newaxis, :]  # (batch_size, 1, seq_len)

class LocalActivationUnit(nn.Module):
    def __init__(self, d_model, middle_units, dropout_rate):
        super(LocalActivationUnit, self).__init__()
        self.d_model = d_model
        self.middle_units = middle_units
        self.dropout_rate = dropout_rate
        self.dnn = nn.Sequential(
            nn.Linear(in_features=middle_units*4, out_features=d_model),
            nn.ReLU(),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.ReLU()
        )

    def forward(self, inputs, training=None, **kwargs):
        query, keys = inputs
        keys_len = keys.shape[1]
        # queries = torch.cat([query] * keys_len, dim=1)
        queries = query.repeat(keys_len ,1 ,1)
        # keys = torch.reshape(keys, (3, -1))
        att_input = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
        att_out = self.dnn(att_input)
        attention_score = nn.Linear(in_features=self.d_model, out_features=1)(att_out)
        return attention_score

# 构造 Din Attention Layer 层

class DinAttentionLayer(nn.Module):
    def __init__(self, d_model, middle_units, dropout_rate):
        super(DinAttentionLayer, self).__init__()
        self.d_model = d_model
        self.middle_units = middle_units
        self.dropout_rate = dropout_rate
        self.local_activation_unit = LocalActivationUnit(d_model, middle_units, dropout_rate)

    def forward(self, inputs, **kwargs):
        query, keys, values, mask = inputs
        scaled_attention_logits = self.local_activation_unit([query, keys])
        scaled_attention_logits = scaled_attention_logits.transpose(1, 2)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, values)
        return output

