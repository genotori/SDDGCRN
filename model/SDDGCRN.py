import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model.GCRNCell import GCRNCell
from model.Layer import nconv, TrendConv, MHAttention, EstimationGate
from lib.util import sym_adj, asym_adj


class GCRN(nn.Module):
    """
    Graph Convolutional Recurrent Network
    """

    def __init__(self, node_num, dim_in, dim_out, embed_dim, cheb_k, num_layers, adpc):
        super(GCRN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.c = adpc
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))
        self.linear = nn.Linear(dim_in, dim_out)
        self.gcn = nconv()

    def forward(self, x, init_state, node_embeddings, adj, ratio):
        # x: (B, T, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        k = self.c + (1 - self.c) * ratio

        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                static_state = self.gcn(current_inputs[:, t, :, :], adj)
                if i == 0:
                    static_state = self.linear(static_state)
                final_state = k * state + (1 - k) * static_state
                inner_states.append(final_state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.hidden_dim = args.rnn_units
        self.batch = args.batch_size
        self.head = args.head
        self.horizon = args.horizon
        self.inh_model = STBlock(args, args.adp_coefficient_inh, args.embed_dim_inh)
        self.dif_model = STBlock(args, args.adp_coefficient_dif, args.embed_dim_dif)
        self.linear1 = nn.Linear(args.input_dim, args.rnn_units, bias=False)
        self.node_embeddings_inh = nn.Parameter(torch.randn(args.num_nodes, args.embed_dim_inh), requires_grad=True)  # 固定信号
        self.node_embeddings_dif = nn.Parameter(torch.randn(args.num_nodes, args.embed_dim_dif), requires_grad=True)  # 扩散信号
        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.embed_dim_inh))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim_inh))
        self.attention = MHAttention(input_channels=args.rnn_units, h=self.head)
        self.conv = nn.Conv2d(in_channels=args.batch_size * args.horizon * self.head, out_channels=1, kernel_size=1)
        self.linear2 = nn.Linear(args.num_nodes, args.num_nodes, bias=False)
        self.estimate_gate = EstimationGate(args.embed_dim_inh, args.embed_dim_dif, args.embed_dim_inh, args.rnn_units)

    def forward(self, source, adj, epoch_idx):
        # source (B, T, N, D)
        adj = torch.Tensor(adj[0]).to(source.device)

        t_i_d_data = source[..., 1]
        T_i_D_emb = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]
        d_i_w_data = source[..., 2]
        D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]

        # 搞一个不同节点之间的关系权重
        source_proj = self.linear1(source)
        simility = self.attention(source_proj)

        adj_attn = self.conv(simility).squeeze()
        new_adj = torch.mul(adj, adj_attn) + self.linear2(adj_attn)
        new_adj = new_adj.fill_diagonal_(0.)
        new_adj = F.softmax(new_adj, dim=-1)
        new_adj = new_adj.fill_diagonal_(1.)
        sym_adj_ = asym_adj(new_adj.detach().cpu().numpy())
        sym_adj_ = torch.Tensor(sym_adj_).to(source.device)

        inh = self.estimate_gate(self.node_embeddings_inh, self.node_embeddings_dif, source[..., :1], T_i_D_emb, D_i_W_emb)
        inh_data = torch.zeros_like(source)
        inh_data[..., :1] = inh
        inh_data[..., 1:] = source[..., 1:]
        inh_output = self.inh_model(inh_data, sym_adj_, epoch_idx, self.node_embeddings_inh)

        dif_data = torch.zeros_like(source)
        dif_data[..., :1] = inh_data[..., :1]
        dif_data = source - dif_data
        dif_output = self.dif_model(dif_data, sym_adj_, epoch_idx, self.node_embeddings_dif)
        return dif_output + inh_output