import torch
import math
from torch import nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):

    def __init__(self, dropout):
        super(DotProductAttention,self).__init__()
        #drop out from pytorch for 
        self.dropout = nn.Dropout(dropout)

    def masked_softmax(self, scores, valid_lens):
        print(scores)
        if valid_lens is not None:
            assert scores.shape[0] == len(valid_lens), "Not the same"

            mask = torch.zeros_like(scores, dtype=torch.bool)

            for i, length in enumerate(valid_lens):
                mask[i,:length] = True
                print(mask)
            print(mask)

            # Apply a very large negative value to the scores at masked positions
            scores.masked_fill_(~mask, float('-inf'))
            return F.softmax(scores, dim=-1)
        else:
            return F.softmax(scores, dim=-1)


    def forward(self, keys, queries, values, valid_lens=None):
        #d = dimension of the hidden state
        d = queries.shape[-1]

        #calculation of attention scores with batch matrix multiplication (bmm, b x m x n X b x p x n)
        scores = torch.bmm(queries,torch.transpose(keys, 1, 2))/ math.sqrt(d) #transpose matrix for multiplication and sqrt(d) for stabilizing variance

        self.weights = self.masked_softmax(scores,valid_lens) #if masking out then valid lens needs to be applied for padding out shorter sequences

        output = torch.bmm(self.dropout(self.weights),values)

        return output
    

queries = torch.normal(0, 1, (2, 1, 2))
keys = torch.normal(0, 1, (2, 10, 2))
values = torch.normal(0, 1, (2, 10, 4))
valid_lens = torch.tensor([2, 6])
attention = DotProductAttention(dropout=0.5)
a = attention(keys=keys, queries=queries, values=values, valid_lens=valid_lens)
print(a)
