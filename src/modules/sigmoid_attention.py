import torch
import torch.nn as nn
import numpy as np

class SigmoidAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, Wq=None, Wk=None, Wv=None):
        super(SigmoidAttention, self).__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads

        self.Wq = Wq if Wq is not None else nn.Linear(feature_dim, feature_dim)
        self.Wk = Wk if Wk is not None else nn.Linear(feature_dim, feature_dim)
        self.Wv = Wv if Wv is not None else nn.Linear(feature_dim, feature_dim)

    def forward(self, x, bias=None):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        attention_scores = torch.matmul(q / np.sqrt(self.feature_dim), k.transpose(-2, -1))

        if bias is not None:
            attention_scores += bias

        attention_weights = torch.sigmoid(attention_scores)
        output = torch.matmul(attention_weights, v)

        return output
