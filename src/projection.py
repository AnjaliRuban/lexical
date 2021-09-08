import torch
from torch import nn
import torch.nn.functional as F
import pdb
from torch.nn.parameter import Parameter
import numpy as np
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
    def __init__(self, src_size, trg_size, embed_dim=512, h_dim=1024, thres=0.5):
        super().__init__()

        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.thres = thres

        self.src_embed = nn.Embedding(src_size, embed_dim)
        self.src_encoder = nn.LSTM(input_size=embed_dim,
                                    hidden_size=h_dim,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)
        self.trg_embed = nn.Embedding(trg_size, embed_dim)
        self.trg_encoder = nn.LSTM(input_size=embed_dim,
                                    hidden_size=h_dim,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)

        # self.lexicon = {}
        self.lexicon_dense = torch.zeros([src_size, trg_size])

    def forward(self, s_unk, s_full, t_unk=None, t_full=None):
        if not t_unk is None:
            s_unk = s_unk.permute(1, 0)
            s_full = s_full.permute(1, 0)
            t_unk = t_unk.permute(1, 0)
            t_full = t_full.permute(1, 0)
            src_size = torch.max(s_full) + 1
            trg_size = torch.max(t_full) + 1
            if src_size >= self.lexicon_sparse.shape[0] or trg_size >= self.lexicon_sparse.shape[1]:
                self.lexicon_sparse = torch.sparse_coo_tensor(self.lexicon_sparse.coalesce().indices(), self.lexicon_sparse.coalesce().values(), size=(max(self.lexicon_sparse.shape[0], src_size), max(self.lexicon_sparse.shape[1], trg_size))).to(s_unk.device)

            src = self.src_embed(s_unk)
            trg = self.trg_embed(t_unk)
            src_h = torch.zeros((2, src.shape[0], self.h_dim)).to(src.device)
            src_c = torch.zeros((2, src.shape[0], self.h_dim)).to(src.device)
            src, (src_h, src_c) = self.src_encoder(src, (src_h, src_c))

            trg_h = torch.zeros((2, trg.shape[0], self.h_dim)).to(src.device)
            trg_c = torch.zeros((2, trg.shape[0], self.h_dim)).to(src.device)
            trg, (trg_h, trg_c) = self.trg_encoder(trg, (trg_h, trg_c))

            cross = torch.sigmoid(torch.bmm(src, trg.permute(0, 2, 1)))

            cross = (cross > self.thres).float()
            cross = (cross == 1.0).nonzero(as_tuple=False)

            row = list(range(cross.shape[0]))

            cross[row, 1] = s_full[0][cross[row, 1]]
            cross[row, 2] = t_full[0][cross[row, 2]]
            # cross = cross[row, 1:]
            cross = cross[row, 1:].permute(1, 0)
            lex = torch.sparse_coo_tensor(cross, [1] * cross.shape[1],[self.lexicon_sparse.shape[0],  self.lexicon_sparse.shape[1]], dtype=torch.float).to(src.device)
            self.lexicon_sparse += lex
            return None
        else:
            embedding = F.embedding(s_full.cpu(), self.lexicon_dense).cuda()
            # embedding = (embedding > 0).float()
            embedding = torch.softmax(embedding, dim=2)
            return embedding

    def ground(self):
        lex = torch.zeros(self.lexicon_sparse.shape)
        lex[:self.lexicon_dense.shape[0], :self.lexicon_dense.shape[1]] = self.lexicon_dense
        lex += self.lexicon_sparse.coalesce().cpu().to_dense()
        self.lexicon_dense = lex
        self.lexicon_sparse = torch.sparse_coo_tensor(torch.empty([2, 0]), [], [self.lexicon_sparse.shape[0], self.lexicon_sparse.shape[1]]).cuda()

    def do_eval(self):
        self.ground()
        del self.lexicon_sparse

    def do_train(self):
        self.lexicon_sparse = torch.sparse_coo_tensor(torch.empty([2, 0]), [], [self.lexicon_dense.shape[0], self.lexicon_dense.shape[1]]).cuda()
