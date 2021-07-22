import torch
from torch import nn
import torch.nn.functional as F
import pdb
from torch.nn.parameter import Parameter
EPS = 1e-7

class SoftAlign(nn.Module):
    def __init__(self, proj, requires_grad=True):
        super().__init__()
        proj_tensor = torch.from_numpy(proj)
        proj_tensor = torch.log(torch.softmax(proj_tensor,dim=1) + EPS)
        self.proj = Parameter(proj_tensor)
        self.proj.requires_grad = requires_grad


    def forward(self, input):
          embedding = torch.softmax(self.proj,dim=1)
          return F.embedding(input, embedding)

class PipelineAlign(nn.Module):
    def __init__(self, src_size, trg_size, embed_dim=512, h_dim=512, thres=0.5):
        super().__init__()

        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.thres = thres

        self.src_embed = nn.Embedding(src_size, embed_dim)
        self.src_encoder = nn.LSTM(input_size=embed_dim,
                                    hidden_dim=h_dim,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)
        self.trg_embed = nn.Embedding(trg_size, embed_size)
        self.trg_encoder = nn.LSTM(input_size=embed_dim,
                                    hidden_dim=h_dim,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)
        
        self.lexicon = torch.zeros((src_size, trg_size))
        
    def forward(self, s, t=None):
        if not t is None:    
            src = self.src_embed(s)
            trg = self.trg_embed(t)

            src_h = torch.zeros((2, src.shape[0], self.h_dim))
            src_c = torch.zeros((2, src.shape[0], self.h_dim))
            src, (src_h, src_c) = self.src_encoder(src, (src_h, src_c))

            trg_h = torch.zeros((2, trg.shape[0], self.h_dim))
            trg_c = torch.zeros((2, trg.shape[0], self.h_dim))
            trg, (trg_h, trg_c) = self.trg_encoder(trg, (trg_h, trg_c))

            cross = torch.sigmoid(torch.bmm(src, trg.permute(0, 2, 1)))
        
            cross = (cross > self.thres).float()
            cross = (cross == 1.0).nonzero(as_tuple=False).tolist()
            for c in cross:
                self.lexicon[s[c[0]]][t[c[1]]] += 1
            return None
        else: 
            embedding = torch.softmax(self.lexicon, dim=1)
            return F.embedding(s, embedding)



