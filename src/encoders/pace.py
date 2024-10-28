import copy
from typing import List, Tuple, Optional, Callable, Dict

import igraph as ig
import numpy as np
import torch
from torch import nn
from torch.nn import LayerNorm, MultiheadAttention
from torch.nn import functional as F

from src.encoders.pace_utils import PaceDag
from src.toolkit.labeled import LABEL_KEY

POSITION_KEY = 'position'


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
            activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
            layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Activation function
        self.activation = activation

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        # Add & Norm
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # Feedforward network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # Add & Norm
        src = src + self.dropout(ff_output)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            encoder_layer: TransformerEncoderLayer,
            num_layers: int,
            norm: Optional[nn.Module] = None
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        output = src

        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask
            )
            if torch.isnan(output).any():
                raise ValueError(f"NaN detected in the output of encoder layer {i}")

        if self.norm:
            output = self.norm(output)

        return output

    @staticmethod
    def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
            layer_norm_eps=1e-5
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout)

        self.activation = activation

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
    ):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        # memory mask to target mask
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=tgt_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        return output


# dag2seq framework
class GnnPositionalEncoding(nn.Module):
    def __init__(self, ninp, dropout, max_n):
        super(GnnPositionalEncoding, self).__init__()
        self.ninp = ninp  # size of the position embedding
        self.max_n = max_n  # maximum position
        self.dropout = dropout
        if dropout > 0.0001:
            self.droplayer = nn.Dropout(p=dropout)
        self.W1 = nn.Parameter(torch.zeros(2 * max_n, 2 * ninp))
        self.W2 = nn.Parameter(torch.zeros(2 * ninp, ninp))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.relu = nn.ReLU()
        self.max_n = max_n

    def forward(self, x, adj):
        """
        x is the postiion list, size = (batch, max_n, max_n): 
        one-hot of position, and nodes after the end type are all zeros embedding
        adj is the adjacency matrix (not the sparse matrix)

        #batch_size = len(x)
        pos_one_hot = torch.zeros(batch_size,self.max_n,self.max_n).to(self._get_device())
        for i in range(batch_size):
        pos_one_hot[i,:len(x[i]),:] = self._one_hot(x[i],self.max_n)
        """
        device = x.device

        pos_embed = torch.cat((x, torch.matmul(adj.transpose(1, 2), x)), 2)  # concat(x_i, sum_j{x_j, j \in N(i)})
        pos_embed = self.relu(torch.matmul(pos_embed, self.W1.to(device)))
        if self.dropout > 0.0001:
            pos_embed = self.droplayer(pos_embed)
        pos_embed = torch.matmul(pos_embed, self.W2.to(device))
        if self.dropout > 0.0001:
            pos_embed = self.droplayer(pos_embed)
        return pos_embed


class PaceVae(nn.Module):
    def __init__(
            self,
            toolkit: PaceDag,
            ninp: int = 256,
            nhead: int = 8,
            nhid: int = 512,
            nlayers: int = 6,
            dropout: float = 0.25,
            fc_hidden: int = 256,
            nz: int = 64,

    ):
        super(PaceVae, self).__init__()

        self.toolkit = toolkit

        self.max_n = self.toolkit.num_vertices  # maximum number of vertices (each node, node type sequence must be 2, 0,.....,1. then we could use all zeros to pad)
        self.nvt = self.toolkit.label_cardinality  # number of vertex types (nvt including the start node type (0), the end node (1), the start sign(2))
        self.graph_label_input = self.toolkit.graph_label_input
        self.graph_label_output = self.toolkit.graph_label_output
        self.graph_label_start = self.toolkit.graph_label_start
        self.ninp = ninp  # size of node type embedding (so as the position embedding)
        self.nhead = nhead  # number of heads in multi-head attention
        self.nhid = nhid  # feedforward network hidden state size (assert nhid = 2 * ninp)
        self.nz = nz  # latent space dimension
        self.nlayers = nlayers  # number pf transformer layers
        if dropout > 0.0001:
            self.droplayer = nn.Dropout(p=dropout)
        self.dropout = dropout
        self.device = None

        self.graph_label_key = self.toolkit.graph_label_key
        self.graph_position_key = self.toolkit.graph_position_key

        # 1. encoder-related  
        self.pos_embed = GnnPositionalEncoding(ninp, dropout, self.max_n)
        self.node_embed = nn.Sequential(
            nn.Linear(self.nvt, ninp),
            nn.ReLU()
        )
        encoder_layers = TransformerEncoderLayer(nhid, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)

        hidden_size = self.nhid * self.max_n
        self.hidden_size = hidden_size
        # self.fc1 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz)) # latent mean
        self.fc1 = nn.Linear(hidden_size, nz)
        self.fc2 = nn.Linear(hidden_size, nz)
        # nn.Linear(hidden_size,nz) 
        # self.fc2 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz))  # latent logvar

        # 2. decoder-related
        decoder_layers = TransformerDecoderLayer(nhid, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers, nlayers)

        self.add_node = nn.Sequential(
            nn.Linear(nhid, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, self.nvt)
        )
        self.add_edge = nn.Sequential(
            nn.Linear(nhid * 2, nhid),
            nn.ReLU(),
            nn.Linear(nhid, 1)
        )  # whether to add edge between v_i and v_new, f(hvi, hnew

        # self.fc3 = nn.Sequential(
        #    nn.Linear(nz,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,hidden_size))
        self.fc3 = nn.Linear(nz, hidden_size)
        # 4. others
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax2 = nn.LogSoftmax(2)
        self.logsoftmax1 = nn.LogSoftmax(1)

    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if not idx:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self._get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self._get_device())
        return x

    def _mask_generate(self, adj, num_node):
        """
        compute the tgt_mask for the decoder. (already been put on the GPU)
        adj type: FloatTensor of the adjacency matrix
        """
        mask = torch.zeros_like(adj).to(self._get_device())
        mem = torch.zeros_like(adj).to(self._get_device())
        ite = 1
        mask += adj
        mem += adj
        while ite <= num_node - 2 and mem.to(torch.uint8).any():
            mem = torch.matmul(mem, adj)
            mask += mem
            # print(ite)
            ite += 1
        del mem
        mask += torch.diag(torch.ones(num_node)).to(self._get_device())
        # mask = mask < 0.5
        # mask = mask.to(torch.bool).t()
        mask = mask < 0.5
        return mask

    def _get_edge_score(self, H):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(H))

    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self._get_device())  # get a zero hidden state

    def _prepare_features(self, pace_graphs_batch: List[ig.Graph], mem_len=None):
        """
        prepare the input node features, adjacency matrix, masks.
        """
        batch_size = len(pace_graphs_batch)
        node_feature = torch.zeros(batch_size, self.max_n, self.nvt).to(
            self._get_device())  # we take one-hot encoding as the initial features
        pos_one_hot = torch.zeros(batch_size, self.max_n, self.max_n).to(self._get_device())  # position encoding
        adj = torch.zeros(batch_size, self.max_n, self.max_n).to(self._get_device())  # adjacency matrix
        src_mask = torch.ones(batch_size * self.nhead, self.max_n - 1, self.max_n - 1).to(
            self._get_device())  # source mask
        # src_mask = torch.zeros(batch_size * self.nhead,self.max_n,self.max_n).to(self._get_device()) # source mask
        tgt_mask = torch.ones(batch_size * self.nhead, self.max_n, self.max_n).to(self._get_device())  # target mask
        mem_mask = torch.ones(batch_size * self.nhead, self.max_n, self.max_n - 1).to(self._get_device())  # target mask

        graph_sizes = []  # number of node in each graph
        true_types = []  # true graph types

        head_count = 0
        for i in range(batch_size):
            g = pace_graphs_batch[i]

            ntype = g.vs[self.graph_label_key]
            ptype = g.vs[self.graph_position_key]

            num_node = len(ntype)
            if num_node < self.max_n:
                ntype += [self.graph_label_output] * (self.max_n - num_node)
                ptype += [max(ptype) + 1] * (self.max_n - num_node)

            # node i feature
            ntype_one_hot = self._one_hot(ntype, self.nvt)
            # the 'extra' nodes are padded with the zero embeddings
            node_feature[i, :, :] = ntype_one_hot
            # position one-hot
            pos_one_hot[i, :, :] = self._one_hot(ptype, self.max_n)
            # node i adj
            adj_i = torch.FloatTensor(g.get_adjacency().data).to(self._get_device())
            adj[i, :num_node, :num_node] = adj_i
            # src mask
            src_mask[head_count:head_count + self.nhead, :num_node - 1, :num_node - 1] = torch.stack(
                [self._mask_generate(adj_i, num_node)[1:, 1:]] * self.nhead, 0)
            # tgt mask
            tgt_mask[head_count:head_count + self.nhead, :num_node, :num_node] = torch.stack(
                [self._mask_generate(adj_i, num_node)] * self.nhead, 0)
            tgt_mask[head_count:head_count + self.nhead, num_node:, num_node:] = torch.zeros(self.nhead,
                                                                                             self.max_n - num_node,
                                                                                             self.max_n - num_node).to(
                self._get_device())
            # memory mask
            if mem_len is None:
                mem_len = num_node - 1
                mem_mask[head_count:head_count + self.nhead, :num_node, :mem_len] = torch.zeros(self.nhead, num_node,
                                                                                                mem_len).to(
                    self._get_device())
                mem_mask[head_count:head_count + self.nhead, num_node:, mem_len:] = torch.zeros(self.nhead,
                                                                                                self.max_n - num_node,
                                                                                                self.max_n - 1 - mem_len).to(
                    self._get_device())
            else:
                mem_mask[head_count:head_count + self.nhead, :num_node, :mem_len] = torch.zeros(self.nhead, num_node,
                                                                                                mem_len).to(
                    self._get_device())
                mem_mask[head_count:head_count + self.nhead, num_node:, -1:] = torch.zeros(self.nhead,
                                                                                           self.max_n - num_node, 1).to(
                    self._get_device())
            # graph size
            # graph size = number of node + 2 (start type and a end type )
            graph_sizes.append(g.vcount())
            # true type
            # we skip the start node for teacher forcing
            true_types.append(g.vs[self.graph_label_key][1:])
            head_count += self.nhead

        return (
            node_feature,
            pos_one_hot,
            adj,
            src_mask.to(torch.bool),
            tgt_mask.to(torch.bool).transpose(1, 2),
            mem_mask.to(torch.bool),
            graph_sizes,
            true_types
        )

    def encode(self, labeled_graphs_batch: List[ig.Graph]) -> Tuple[torch.Tensor, torch.Tensor]:

        pace_graphs_batch = [self.toolkit.from_labeled_graph_to_graph(graph) for graph in labeled_graphs_batch]

        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(
            pace_graphs_batch)

        pos_feat = self.pos_embed(pos_one_hot, adj)
        node_feat = self.node_embed(node_one_hot)
        node_feat = torch.cat([node_feat, pos_feat], 2)

        # here we set the source sequence and the tgt sequence for the teacher forcing
        # node 2 is the start symbol, shape: (bsiaze, max_n-1, nhid)
        src_inp = node_feat.transpose(0, 1)

        # memory = self.encoder(src_inp,mask=src_mask)
        memory = self.encoder(src_inp, mask=tgt_mask)
        # shape ( batch_size, self.max_n-1, nhid): each batch, the first num_node - 1 rows are the representation of input nodes.
        memory = memory.transpose(0, 1).reshape(-1, self.max_n * self.nhid)

        return self.fc1(memory), self.fc2(memory)

    def reparameterize(
            self,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            eps_scale: float = 0.01
    ) -> torch.Tensor:
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z: torch.Tensor) -> List[ig.Graph]:
        """
        This is a sequence to sequence prediction model.
        Input: a graph (sequence of nodes)
        from a graph_label_input node, we use the transformer to predict the type of the next node
        and this process is continued until the graph_label_output node (or iterations reaches max_n)
        """
        batch_size = len(z)
        memory = self.fc3(z).reshape(-1, self.max_n, self.nhid).transpose(0, 1)

        pace_graphs_batch = [ig.Graph(directed=True) for _ in range(batch_size)]
        for g in pace_graphs_batch:
            g.add_vertex()
            g.vs[0][self.graph_label_key] = self.graph_label_start
            g.vs[0][self.graph_position_key] = 0

            g.add_vertex()
            g.vs[1][self.graph_label_key] = self.graph_label_input
            g.vs[1][self.graph_position_key] = 1

        # memory = self.encoder(src_inp,mask=src_mask)

        finished = [False] * batch_size
        # the first two type of nodes are certain
        for idx in range(2, self.max_n):
            node_one_hot, pos_one_hot, adj, _, tgt_mask, mem_mask, _, _ = self._prepare_features(pace_graphs_batch,
                                                                                                 self.max_n - 1)
            pos_feat = self.pos_embed(pos_one_hot, adj)
            node_feat = self.node_embed(node_one_hot)
            node_feat = torch.cat([node_feat, pos_feat], 2)
            tgt_inp = node_feat.transpose(0, 1)

            out = self.decoder(tgt_inp, memory, tgt_mask=tgt_mask, memory_mask=mem_mask)
            out = out.transpose(0, 1)  # shape ( batch_size, self.max_n, nvrt)
            next_node_hidden = out[:, idx - 1, :]
            # add nodes
            type_scores = self.add_node(next_node_hidden)
            type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
            new_types = [np.random.choice(range(self.nvt), p=type_probs[i]) for i in range(len(pace_graphs_batch))]
            # add edges
            # just from the cneter node to the target node
            edge_scores = torch.cat([torch.stack([next_node_hidden] * (idx - 1), 1), out[:, :idx - 1, :]], -1)
            edge_scores = self._get_edge_score(edge_scores)

            for i, g in enumerate(pace_graphs_batch):
                if not finished[i]:
                    if idx < self.max_n - 1:
                        g.add_vertex(type=new_types[i])
                    else:
                        g.add_vertex(type=self.graph_label_output)
            for vi in range(idx - 2, -1, -1):
                ei_score = edge_scores[:, vi]  # 0 point to node 1
                random_score = torch.rand_like(ei_score)
                decisions = random_score < ei_score
                for i, g in enumerate(pace_graphs_batch):
                    if finished[i]:
                        continue
                    if new_types[i] == self.graph_label_output:
                        # if new node is graph_label_output, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0)
                                            if v.index != g.vcount() - 1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount() - 1)
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi + 1, g.vcount() - 1)

            for pace_graph in pace_graphs_batch:
                pace_graph.vs[self.graph_position_key] = self.toolkit.compute_graph_positions(pace_graph)

        labeled_graphs_batch = [self.toolkit.from_graph_to_labeled_graph(pace_graph) for pace_graph in
                                pace_graphs_batch]

        return labeled_graphs_batch

    def loss(
            self,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            labeled_graphs_batch: List[ig.Graph],
            beta: float = 0.005
    ):

        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct
        # (batch_size, hidden)
        z = self.reparameterize(mu, logvar)  # (batch_size, hidden)
        memory = self.fc3(z).reshape(-1, self.max_n, self.nhid).transpose(0, 1)

        pace_graphs_batch = [self.toolkit.from_labeled_graph_to_graph(graph) for graph in labeled_graphs_batch]

        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(
            pace_graphs_batch)
        batch_size = len(graph_sizes)
        pos_feat = self.pos_embed(pos_one_hot, adj)
        node_feat = self.node_embed(node_one_hot)
        node_feat = torch.cat([node_feat, pos_feat], 2)

        tgt_inp = node_feat.transpose(0, 1)
        # shape (self.max_n, batch_size, nhid)
        out = self.decoder(tgt_inp, memory, tgt_mask=tgt_mask, memory_mask=mem_mask)
        out = out.transpose(0, 1)

        scores = self.add_node(out)
        # shape ( batch_size, self.max_n, nvrt)
        scores = self.logsoftmax2(scores)
        # loglikelihood
        res = 0
        for i in range(batch_size):
            # vertex log likelihood
            # print(true_types[i])
            if len(true_types[i]) < self.max_n:
                true_types[i] += [0] * (self.max_n - len(true_types[i]))
            # only count 'no padding' nodes. graph size i - 1 since the input symbol of the encoder do not have the start node
            vll = scores[i][np.arange(self.max_n), true_types[i]][:graph_sizes[i] - 1].sum()
            res += vll
            # edges log likelihood
            # no start node
            num_node_i = graph_sizes[i] - 1
            num_pot_edges = int(num_node_i * (num_node_i - 1) / 2.0)
            edge_scores = torch.zeros(num_pot_edges, 2 * self.nhid).to(self._get_device())
            ground_truth = torch.zeros(num_pot_edges, 1).to(self._get_device())
            count = 0
            for idx in range(num_node_i - 1, 0, -1):
                # in each batch, ith row of out represent the presentation of node i+1 (since input do not have the start node)
                edge_scores[count:count + idx, :] = torch.cat([torch.stack([out[i, idx, :]] * idx, 0), out[i, :idx, :]],
                                                              -1)
                ground_truth[count:count + idx, :] = adj[i, 1:idx + 1, idx + 1].view(idx, 1)
                count += idx

            edge_scores = self._get_edge_score(edge_scores)
            ell = - F.binary_cross_entropy(edge_scores, ground_truth, reduction='sum')
            res += ell

        # convert likelihood to loss
        res = -res
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta * kld, res, kld

    def encode_decode(self, labeled_graphs_batch: List[ig.Graph]) -> List[ig.Graph]:
        mu, logvar = self.encode(labeled_graphs_batch)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self):
        pass


class PaceVaeV2(nn.Module):
    def __init__(
            self,
            toolkit: PaceDag,
            ninp: int = 256,
            nhead: int = 8,
            nhid: int = 512,
            nlayers: int = 6,
            dropout: float = 0.25,
            fc_hidden: int = 256,
            nz: int = 64,
    ):
        super(PaceVaeV2, self).__init__()

        self.toolkit = toolkit

        self.max_num_vertices = self.toolkit.num_vertices  # maximum number of vertices (each node, node type sequence must be 2, 0,.....,1. then we could use all zeros to pad)
        self.vertex_label_cardinality = self.toolkit.label_cardinality  # number of vertex types (nvt including the start node type (0), the end node (1), the start sign(2))
        self.graph_label_input = self.toolkit.graph_label_input
        self.graph_label_output = self.toolkit.graph_label_output
        self.graph_label_start = self.toolkit.graph_label_start
        self.ninp = ninp  # size of node type embedding (so as the position embedding)
        self.nhead = nhead  # number of heads in multi-head attention
        self.nhid = nhid  # feedforward network hidden state size (assert nhid = 2 * ninp)
        self.latent_dim = nz  # latent space dimension
        self.nlayers = nlayers  # number pf transformer layers
        if dropout > 0.0001:
            self.droplayer = nn.Dropout(p=dropout)
        self.dropout = dropout
        self.device = None

        self.graph_label_key = self.toolkit.graph_label_key
        self.graph_position_key = self.toolkit.graph_position_key

        # 1. encoder-related
        self.vertex_position_embed = GnnPositionalEncoding(ninp, dropout, self.max_num_vertices)
        self.vertex_label_embed = nn.Sequential(
            nn.Linear(self.vertex_label_cardinality, ninp),
            nn.ReLU()
        )
        encoder_layers = TransformerEncoderLayer(nhid, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)

        hidden_size = self.nhid * self.max_num_vertices
        self.hidden_size = hidden_size
        # self.fc1 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz)) # latent mean
        self.fc1 = nn.Linear(hidden_size, nz)
        self.fc2 = nn.Linear(hidden_size, nz)
        # nn.Linear(hidden_size,nz)
        # self.fc2 = nn.Sequential(
        #    nn.Linear(hidden_size,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,nz))  # latent logvar

        # 2. decoder-related
        decoder_layers = TransformerDecoderLayer(nhid, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers, nlayers)

        self.add_node = nn.Sequential(
            nn.Linear(nhid, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, self.vertex_label_cardinality)
        )
        self.add_edge = nn.Sequential(
            nn.Linear(nhid * 2, nhid),
            nn.ReLU(),
            nn.Linear(nhid, 1)
        )  # whether to add edge between v_i and v_new, f(hvi, hnew

        # self.fc3 = nn.Sequential(
        #    nn.Linear(nz,2*nz),
        #    nn.ReLU(),
        #    nn.Linear(2*nz,hidden_size))
        self.fc3 = nn.Linear(nz, hidden_size)
        # 4. others
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax2 = nn.LogSoftmax(2)
        self.logsoftmax1 = nn.LogSoftmax(1)

    def _get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    @staticmethod
    def generate_mask(
            adjacency_matrix: torch.Tensor,
            num_nodes: int,
            device
    ) -> torch.Tensor:
        """
        Generates a mask of non-reachable nodes in a graph represented by the adjacency matrix.
        """
        reachable_mask = adjacency_matrix.clone().to(torch.bool)
        current_reachability = adjacency_matrix.clone()

        for _ in range(1, num_nodes - 1):
            # Compute new reachability using integer matrix multiplication for efficiency
            new_reachability = torch.matmul(
                current_reachability.to(torch.int),
                adjacency_matrix.to(torch.int)
            ).to(torch.bool)

            # Identify new nodes that have not been reached before
            new_reachability &= ~reachable_mask

            if not new_reachability.any():
                break

            # Update reachable nodes and set current reach for next iteration
            reachable_mask |= new_reachability
            current_reachability = new_reachability

        # Include self-reachability
        reachable_mask |= torch.eye(num_nodes, dtype=torch.bool, device=device)

        # Non-reachable mask is the inverse of reachable_mask
        non_reachable_mask = ~reachable_mask

        return non_reachable_mask

    def prepare_features(
            self,
            pace_graphs_batch: List[ig.Graph],
            device: torch.device,
            fixed_memory_len: Optional[int] = None,
    ):

        batch_size = len(pace_graphs_batch)

        # Initialize feature tensors directly on the target device
        vertex_label_features = torch.zeros(
            batch_size,
            self.max_num_vertices,
            self.vertex_label_cardinality,
            device=device,
        )
        vertex_position_features = torch.zeros(
            batch_size,
            self.max_num_vertices,
            self.max_num_vertices,
            device=device,
        )
        adjacency_matrices = torch.zeros(
            batch_size,
            self.max_num_vertices,
            self.max_num_vertices,
            device=device,
        )

        # Initialize masks
        total_heads = batch_size * self.nhead
        source_masks = torch.ones(
            total_heads,
            self.max_num_vertices - 1,
            self.max_num_vertices - 1,
            device=device,
        )
        target_masks = torch.ones(
            batch_size * self.nhead,
            self.max_num_vertices,
            self.max_num_vertices,
            device=device,
        )
        memory_masks = torch.ones(
            batch_size * self.nhead,
            self.max_num_vertices,
            self.max_num_vertices - 1,
            device=device,
        )

        head_offset = 0

        for i, graph in enumerate(pace_graphs_batch):
            vertex_labels_i = graph.vs[self.graph_label_key]
            vertex_positions_i = graph.vs[self.graph_position_key]

            num_vertices_i = len(vertex_labels_i)

            # Handle padding
            pad_length = self.max_num_vertices - num_vertices_i
            if pad_length > 0:
                vertex_labels_i += [self.graph_label_output] * pad_length
                vertex_positions_i += [max(vertex_positions_i) + 1] * pad_length

            # One-hot encoding
            vertex_label_features[i, :, :] = F.one_hot(
                torch.tensor(vertex_labels_i, dtype=torch.long, device=device),
                num_classes=self.vertex_label_cardinality,
            )
            vertex_position_features[i, :, :] = F.one_hot(
                torch.tensor(vertex_positions_i, dtype=torch.long, device=device),
                num_classes=self.max_num_vertices,
            )

            # Adjacency matrix
            adj = torch.tensor(
                graph.get_adjacency().data,
                dtype=torch.float32,
                device=device,
            )
            adjacency_matrices[i, :num_vertices_i, :num_vertices_i] = adj

            source_masks[head_offset:head_offset + self.nhead, :num_vertices_i - 1, :num_vertices_i - 1] = torch.stack(
                [self.generate_mask(adj, num_vertices_i, device)[1:, 1:]] * self.nhead,
                dim=0,
            )

            target_masks[head_offset:head_offset + self.nhead, :num_vertices_i, :num_vertices_i] = torch.stack(
                [self.generate_mask(adj, num_vertices_i, device)] * self.nhead,
                dim=0,
            )
            target_masks[head_offset:head_offset + self.nhead, num_vertices_i:, num_vertices_i:] = torch.zeros(
                self.nhead,
                self.max_num_vertices - num_vertices_i,
                self.max_num_vertices - num_vertices_i,
                device=device,
            )

            memory_len = num_vertices_i - 1 if fixed_memory_len is None else fixed_memory_len

            memory_masks[head_offset:head_offset + self.nhead, :num_vertices_i, :memory_len] = torch.zeros(
                self.nhead,
                num_vertices_i,
                memory_len,
                device=device,
            )
            memory_masks[head_offset:head_offset + self.nhead, num_vertices_i:, -1:] = torch.zeros(
                self.nhead,
                self.max_num_vertices - num_vertices_i,
                1,
                device=device,
            )

            head_offset += self.nhead

        # We skip the start node for teacher forcing
        vertex_labels = [graph.vs[self.graph_label_key][1:] for graph in pace_graphs_batch]

        # Number of nodes
        num_vertices = [graph.vcount() for graph in pace_graphs_batch]

        return {
            "vertex_label_features": vertex_label_features,
            "vertex_position_features": vertex_position_features,
            "adjacency_matrices": adjacency_matrices,
            "source_masks": source_masks.to(torch.bool),
            "target_masks": target_masks.to(torch.bool).transpose(1, 2),
            "memory_masks": memory_masks.to(torch.bool),
            "num_vertices": num_vertices,
            "vertex_labels": vertex_labels
        }

    def encode(self, labeled_graphs_batch: List[ig.Graph]) -> Tuple[torch.Tensor, torch.Tensor]:

        device = self._get_device()

        pace_graphs_batch = [self.toolkit.from_labeled_graph_to_graph(graph) for graph in labeled_graphs_batch]

        features = self.prepare_features(pace_graphs_batch, device)

        vertex_label_features = features["vertex_label_features"]
        vertex_position_features = features["vertex_position_features"]
        adjacency_matrices = features["adjacency_matrices"]
        target_masks = features["target_masks"]

        vertex_position_embeddings = self.vertex_position_embed(vertex_position_features, adjacency_matrices)
        vertex_label_embeddings = self.vertex_label_embed(vertex_label_features)

        vertex_features = torch.cat(
            [
                vertex_label_embeddings,
                vertex_position_embeddings
            ],
            2
        )

        # here we set the source sequence and the tgt sequence for the teacher forcing
        # node 2 is the start symbol, shape: (bsiaze, max_n-1, nhid)
        src_inp = vertex_features.transpose(0, 1)

        # memory = self.encoder(src_inp,mask=src_mask)
        memory = self.encoder(src_inp, mask=target_masks)
        # shape ( batch_size, self.max_n-1, nhid): each batch, the first num_node - 1 rows are the representation of input nodes.
        memory = memory.transpose(0, 1).reshape(-1, self.max_num_vertices * self.nhid)

        return self.fc1(memory), self.fc2(memory)

    def reparameterize(
            self,
            mu: torch.Tensor,
            log_var: torch.Tensor,
            epsilon_scale: float = 0.01
    ) -> torch.Tensor:
        """
        Performs the reparameterization trick to sample from N(mu, std) during training.
        During evaluation, returns the mean value mu.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std) * epsilon_scale
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> List[ig.Graph]:
        """
        This is a sequence to sequence prediction model.
        Input: a graph (sequence of nodes)
        from a graph_label_input node, we use the transformer to predict the type of the next node
        and this process is continued until the graph_label_output node (or iterations reaches max_n)
        """
        device = self._get_device()

        batch_size = len(z)
        memory = self.fc3(z).reshape(-1, self.max_num_vertices, self.nhid).transpose(0, 1)

        pace_graphs_batch = [ig.Graph(directed=True) for _ in range(batch_size)]
        for g in pace_graphs_batch:
            g.add_vertex()
            g.vs[0][self.graph_label_key] = self.graph_label_start
            g.vs[0][self.graph_position_key] = 0

            g.add_vertex()
            g.vs[1][self.graph_label_key] = self.graph_label_input
            g.vs[1][self.graph_position_key] = 1

        # memory = self.encoder(src_inp,mask=src_mask)

        finished = [False] * batch_size
        # the first two type of nodes are certain
        for idx in range(2, self.max_num_vertices):
            features = self.prepare_features(pace_graphs_batch, device, self.max_num_vertices - 1)

            vertex_label_features = features["vertex_label_features"]
            vertex_position_features = features["vertex_position_features"]
            adjacency_matrices = features["adjacency_matrices"]
            target_masks = features["target_masks"]
            memory_masks = features["memory_masks"]

            pos_feat = self.vertex_position_embed(vertex_position_features, adjacency_matrices)
            node_feat = self.vertex_label_embed(vertex_label_features)
            node_feat = torch.cat([node_feat, pos_feat], 2)
            tgt_inp = node_feat.transpose(0, 1)

            out = self.decoder(tgt_inp, memory, tgt_mask=target_masks, memory_mask=memory_masks)
            out = out.transpose(0, 1)  # shape ( batch_size, self.max_n, nvrt)
            next_node_hidden = out[:, idx - 1, :]
            # add nodes
            type_scores = self.add_node(next_node_hidden)
            type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
            new_types = [np.random.choice(range(self.vertex_label_cardinality), p=type_probs[i]) for i in
                         range(len(pace_graphs_batch))]
            # add edges
            # just from the cneter node to the target node
            edge_scores = torch.cat([torch.stack([next_node_hidden] * (idx - 1), 1), out[:, :idx - 1, :]], -1)
            edge_scores = self.sigmoid(self.add_edge(edge_scores))

            for i, g in enumerate(pace_graphs_batch):
                if not finished[i]:
                    if idx < self.max_num_vertices - 1:
                        g.add_vertex(type=new_types[i])
                    else:
                        g.add_vertex(type=self.graph_label_output)
            for vi in range(idx - 2, -1, -1):
                ei_score = edge_scores[:, vi]  # 0 point to node 1
                random_score = torch.rand_like(ei_score)
                decisions = random_score < ei_score
                for i, g in enumerate(pace_graphs_batch):
                    if finished[i]:
                        continue
                    if new_types[i] == self.graph_label_output:
                        # if new node is graph_label_output, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0)
                                            if v.index != g.vcount() - 1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount() - 1)
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi + 1, g.vcount() - 1)

            for pace_graph in pace_graphs_batch:
                pace_graph.vs[self.graph_position_key] = self.toolkit.compute_graph_positions(pace_graph)

        labeled_graphs_batch = [self.toolkit.from_graph_to_labeled_graph(pace_graph) for pace_graph in
                                pace_graphs_batch]

        return labeled_graphs_batch

    def loss(
            self,
            mu: torch.Tensor,
            log_var: torch.Tensor,
            labeled_graphs_batch: List[ig.Graph],
            beta: float = 0.005
    ):
        device = self._get_device()

        # Reparameterization trick
        latent_vectors = self.reparameterize(mu, log_var)

        # Transform latent vector for decoder
        decoder_states = self.fc3(latent_vectors).reshape(-1, self.max_num_vertices, self.nhid).transpose(0, 1)

        pace_graphs_batch = [self.toolkit.from_labeled_graph_to_graph(graph) for graph in labeled_graphs_batch]

        # Prepare features
        features = self.prepare_features(pace_graphs_batch, device)

        vertex_label_features = features["vertex_label_features"]
        vertex_position_features = features["vertex_position_features"]
        adjacency_matrices = features[
            "adjacency_matrices"]  # adjacency_matrices: [batch_size, max_num_vertices, max_num_vertices]
        target_masks = features["target_masks"]
        memory_masks = features["memory_masks"]
        num_vertices = features["num_vertices"]
        vertex_labels = features["vertex_labels"]

        batch_size = len(num_vertices)

        # Embeddings
        vertex_position_embeddings = self.vertex_position_embed(vertex_position_features, adjacency_matrices)
        vertex_label_embeddings = self.vertex_label_embed(vertex_label_features)

        vertex_features = torch.cat(
            [
                vertex_label_embeddings,
                vertex_position_embeddings,
            ],
            dim=2,
        )

        # Decoder input
        decoder_input = vertex_features.transpose(0, 1)

        # Decode
        decoder_output = self.decoder(
            decoder_input,
            decoder_states,
            tgt_mask=target_masks,
            memory_mask=memory_masks
        ).transpose(0, 1)  # Shape: [batch_size, max_num_vertices, nhid]

        # Node predictions
        node_scores = self.add_node(decoder_output)  # Shape: [batch_size, max_num_vertices, num_node_classes]
        node_log_probs = self.logsoftmax2(node_scores)  # Shape: [batch_size, max_num_vertices, num_node_classes]

        # Initialize log likelihood
        log_likelihood = torch.tensor(0.0, device=device)

        padded_true_node_types = torch.zeros(batch_size, self.max_num_vertices, dtype=torch.long, device=device)
        for i, types in enumerate(vertex_labels):
            length = min(len(types), self.max_num_vertices)
            padded_true_node_types[i, :length] = torch.tensor(types[:length], device=device)

        # Vectorized computation of node log likelihood
        graph_sizes_tensor = torch.tensor(num_vertices, device=device)
        valid_node_mask = torch.arange(self.max_num_vertices, device=device).expand(batch_size,
                                                                                    self.max_num_vertices) < (
                                  graph_sizes_tensor - 1).unsqueeze(1)

        indices = padded_true_node_types.unsqueeze(2)  # Shape: [batch_size, max_num_vertices, 1]

        true_log_probs = torch.gather(node_log_probs, 2, indices).squeeze(2)  # Shape: [batch_size, max_num_vertices]

        masked_true_log_probs = true_log_probs * valid_node_mask.float()  # Shape: [batch_size, max_num_vertices]

        log_likelihood += masked_true_log_probs.sum()

        for i in range(batch_size):

            num_node_i = num_vertices[i] - 1
            num_potential_edges = int(num_node_i * (num_node_i - 1) / 2.0)
            edge_scores = torch.zeros(num_potential_edges, 2 * self.nhid,
                                      device=device)  # Shape: [num_potential_edges, 2 * nhid]
            ground_truth = torch.zeros(num_potential_edges, 1, device=device)  # Shape: [num_potential_edges, 1]
            count = 0

            for idx in range(num_node_i - 1, 0, -1):
                repeated_output = decoder_output[i, idx].unsqueeze(0).repeat(idx, 1)  # Shape: [idx, nhid]
                connecting_output = decoder_output[i, :idx]  # Shape: [idx, nhid]
                edge_scores[count:count + idx, :] = torch.cat([repeated_output, connecting_output], dim=1)

                ground_truth[count:count + idx, :] = adjacency_matrices[i, 1:idx + 1, idx + 1].view(idx, 1)
                count += idx

            # Compute edge log-likelihood
            edge_log_likelihood = - F.binary_cross_entropy_with_logits(self.add_edge(edge_scores), ground_truth,
                                                                       reduction='sum')
            log_likelihood += edge_log_likelihood

        # KL Divergence
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        total_loss = -log_likelihood + beta * kl_divergence

        return total_loss, -log_likelihood, kl_divergence

    def encode_decode(self, labeled_graphs_batch: List[ig.Graph]) -> List[ig.Graph]:
        mu, logvar = self.encode(labeled_graphs_batch)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self):
        pass


class PaceVaeV3(nn.Module):
    def __init__(
            self,
            max_num_vertices: int,
            vertex_label_cardinality: int,
            vertices_embedding_size: int = 256,
            num_heads: int = 8,
            num_layers: int = 6,
            ff_hidden_size: int = 512,
            latent_layer_size: int = 64,
            fc_hidden: int = 256,
            dropout: float = 0.25,
            graph_label_key: str = LABEL_KEY,
            graph_position_key: str = POSITION_KEY,
            graph_label_input: int = 0,
            graph_label_output: int = 1,
            graph_label_start: int = 2
    ):
        super(PaceVaeV3, self).__init__()

        self._max_num_vertices = max_num_vertices + 3
        self._vertex_label_cardinality = vertex_label_cardinality + 3

        self.vertices_embedding_size = vertices_embedding_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_hidden_size = ff_hidden_size
        self.latent_layer_size = latent_layer_size
        self.dropout = dropout

        self._graph_label_key = graph_label_key
        self._graph_position_key = graph_position_key
        self._graph_label_input = graph_label_input
        self._graph_label_output = graph_label_output
        self._graph_label_start = graph_label_start


        self.vertex_position_embed = GnnPositionalEncoding(
            vertices_embedding_size,
            dropout,
            self.max_num_vertices,
        )
        self.vertex_label_embed = nn.Sequential(
            nn.Linear(self._vertex_label_cardinality, vertices_embedding_size),
            nn.ReLU()
        )
        encoder_layers = TransformerEncoderLayer(ff_hidden_size, num_heads, ff_hidden_size, dropout)
        self.encoder = TransformerEncoder(encoder_layers, num_layers)

        hidden_size = self.ff_hidden_size * self.max_num_vertices
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, latent_layer_size)
        self.fc2 = nn.Linear(hidden_size, latent_layer_size)

        decoder_layers = TransformerDecoderLayer(ff_hidden_size, num_heads, ff_hidden_size, dropout)
        self.decoder = TransformerDecoder(decoder_layers, num_layers)

        self.add_node = nn.Sequential(
            nn.Linear(ff_hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, self._vertex_label_cardinality)
        )
        self.add_edge = nn.Sequential(
            nn.Linear(ff_hidden_size * 2, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, 1)
        )

        self.fc3 = nn.Linear(latent_layer_size, hidden_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.logsoftmax1 = nn.LogSoftmax(1)
        self.logsoftmax2 = nn.LogSoftmax(2)


    @property
    def max_num_vertices(self) -> int:
        return self._max_num_vertices

    @property
    def vertex_label_cardinality(self) -> int:
        return self._vertex_label_cardinality

    @property
    def graph_label_key(self) -> str:
        return self._graph_label_key

    @property
    def graph_position_key(self) -> str:
        return self._graph_position_key

    @property
    def graph_label_input(self) -> int:
        return self._graph_label_input

    @property
    def graph_label_output(self) -> int:
        return self._graph_label_output

    @property
    def graph_label_start(self) -> int:
        return self._graph_label_start

    @staticmethod
    def compute_graph_positions(graph: ig.Graph) -> List[int]:
        graph_positions = graph.topological_sorting()
        return graph_positions

    def from_labeled_graph_to_pace_graph(self, labeled_graph: ig.Graph) -> ig.Graph:
        assert self.max_num_vertices - 3 == labeled_graph.vcount(), f"Expected {self.max_num_vertices - 3}, got instead {labeled_graph.vcount()}"

        graph = ig.Graph(directed=True)
        graph.add_vertices(self.max_num_vertices)

        # Set start, input, output vertex types
        graph.vs[0][self.graph_label_key] = self.graph_label_start
        graph.vs[1][self.graph_label_key] = self.graph_label_input

        output_vertex_id = self.max_num_vertices - 1
        graph.vs[output_vertex_id][self.graph_label_key] = self.graph_label_output

        # Add edges from start to input
        graph.add_edge(0, 1)

        for vertex_id in range(self.max_num_vertices - 3):
            vertex: ig.Vertex = labeled_graph.vs[vertex_id]

            vertex_label = vertex[self.graph_label_key] + 3
            graph.vs[vertex_id + 2][self.graph_label_key] = vertex_label

            vertex_connections = [(v.index + 2, vertex_id + 2) for v in vertex.predecessors()]

            if len(vertex_connections) == 0:
                graph.add_edge(1, vertex_id + 2)
                continue

            graph.add_edges(vertex_connections)

        # Add edge from last nodes to output
        end_vertices = [vertex.index for vertex in graph.vs.select(_outdegree_eq=0) if vertex.index != output_vertex_id]

        for vertex_id in end_vertices:
            graph.add_edge(vertex_id, output_vertex_id)

        graph.vs()[POSITION_KEY] = self.compute_graph_positions(graph)

        return graph

    def from_pace_graph_to_labeled_graph(self, graph: ig.Graph) -> ig.Graph:

        labeled_graph = ig.Graph(directed=True)
        labeled_graph.add_vertices(self.max_num_vertices - 3)

        for vertex_id in range(2, self.max_num_vertices - 1):
            labeled_graph.vs[vertex_id - 2][self.graph_label_key] = graph.vs[vertex_id][self.graph_label_key] - 3

            if vertex_id == self.graph_label_start:
                continue

            for incoming_vertex_id in range(2, vertex_id + 2):
                if graph.are_adjacent(incoming_vertex_id, vertex_id):
                    labeled_graph.add_edge(incoming_vertex_id - 2, vertex_id - 2)

        return labeled_graph

    def generate_mask(
            self,
            adjacency_matrix: torch.Tensor,
            num_nodes: int,
    ) -> torch.Tensor:
        """
        Generates a mask of non-reachable nodes in a graph represented by the adjacency matrix.
        """
        device = next(self.parameters()).device

        reachable_mask = adjacency_matrix.clone().to(torch.bool)
        current_reachability = adjacency_matrix.clone()

        for _ in range(1, num_nodes - 1):
            # Compute new reachability using integer matrix multiplication for efficiency
            new_reachability = torch.matmul(
                current_reachability.to(torch.int),
                adjacency_matrix.to(torch.int)
            ).to(torch.bool)

            # Identify new nodes that have not been reached before
            new_reachability &= ~reachable_mask

            if not new_reachability.any():
                break

            # Update reachable nodes and set current reach for next iteration
            reachable_mask |= new_reachability
            current_reachability = new_reachability

        # Include self-reachability
        reachable_mask |= torch.eye(num_nodes, dtype=torch.bool, device=device)

        # Non-reachable mask is the inverse of reachable_mask
        non_reachable_mask = ~reachable_mask

        return non_reachable_mask

    def prepare_features(
            self,
            labeled_graphs_batch: List[ig.Graph],
            fixed_memory_len: Optional[int] = None,
    ):
        device = next(self.parameters()).device

        pace_graphs_batch = [self.from_labeled_graph_to_pace_graph(graph) for graph in labeled_graphs_batch]

        batch_size = len(pace_graphs_batch)

        # Initialize feature tensors directly on the target device
        vertex_label_features = torch.zeros(
            batch_size,
            self.max_num_vertices,
            self.vertex_label_cardinality,
            device=device,
        )
        vertex_position_features = torch.zeros(
            batch_size,
            self.max_num_vertices,
            self.max_num_vertices,
            device=device,
        )
        adjacency_matrices = torch.zeros(
            batch_size,
            self.max_num_vertices,
            self.max_num_vertices,
            device=device,
        )

        # Initialize masks
        total_heads = batch_size * self.num_heads
        source_masks = torch.ones(
            total_heads,
            self.max_num_vertices - 1,
            self.max_num_vertices - 1,
            device=device,
        )
        target_masks = torch.ones(
            batch_size * self.num_heads,
            self.max_num_vertices,
            self.max_num_vertices,
            device=device,
        )
        memory_masks = torch.ones(
            batch_size * self.num_heads,
            self.max_num_vertices,
            self.max_num_vertices - 1,
            device=device,
        )

        head_offset = 0

        for i, graph in enumerate(pace_graphs_batch):
            vertex_labels_i = graph.vs[self.graph_label_key]
            vertex_positions_i = graph.vs[self.graph_position_key]

            num_vertices_i = len(vertex_labels_i)

            # Handle padding
            pad_length = self.max_num_vertices - num_vertices_i
            if pad_length > 0:
                vertex_labels_i += [self.graph_label_output] * pad_length
                vertex_positions_i += [max(vertex_positions_i) + 1] * pad_length

            # One-hot encoding
            vertex_label_features[i, :, :] = F.one_hot(
                torch.tensor(vertex_labels_i, dtype=torch.long, device=device),
                num_classes=self.vertex_label_cardinality,
            )
            vertex_position_features[i, :, :] = F.one_hot(
                torch.tensor(vertex_positions_i, dtype=torch.long, device=device),
                num_classes=self.max_num_vertices,
            )

            # Adjacency matrix
            adj = torch.tensor(
                graph.get_adjacency().data,
                dtype=torch.float,
                device=device,
            )
            adjacency_matrices[i, :num_vertices_i, :num_vertices_i] = adj

            source_masks[head_offset:head_offset + self.num_heads, :num_vertices_i - 1,
            :num_vertices_i - 1] = torch.stack(
                [self.generate_mask(adj, num_vertices_i)[1:, 1:]] * self.num_heads,
                dim=0,
            )

            target_masks[head_offset:head_offset + self.num_heads, :num_vertices_i, :num_vertices_i] = torch.stack(
                [self.generate_mask(adj, num_vertices_i)] * self.num_heads,
                dim=0,
            )
            target_masks[head_offset:head_offset + self.num_heads, num_vertices_i:, num_vertices_i:] = torch.zeros(
                self.num_heads,
                self.max_num_vertices - num_vertices_i,
                self.max_num_vertices - num_vertices_i,
                device=device,
            )

            memory_len = num_vertices_i - 1 if fixed_memory_len is None else fixed_memory_len

            memory_masks[head_offset:head_offset + self.num_heads, :num_vertices_i, :memory_len] = torch.zeros(
                self.num_heads,
                num_vertices_i,
                memory_len,
                device=device,
            )
            memory_masks[head_offset:head_offset + self.num_heads, num_vertices_i:, -1:] = torch.zeros(
                self.num_heads,
                self.max_num_vertices - num_vertices_i,
                1,
                device=device,
            )

            head_offset += self.num_heads

        # We skip the start node for teacher forcing
        vertex_labels = [graph.vs[self.graph_label_key][1:] for graph in pace_graphs_batch]

        # Number of nodes
        num_vertices = [graph.vcount() for graph in pace_graphs_batch]

        return {
            "vertex_label_features": vertex_label_features,
            "vertex_position_features": vertex_position_features,
            "adjacency_matrices": adjacency_matrices,
            "source_masks": source_masks.to(torch.bool),
            "target_masks": target_masks.to(torch.bool).transpose(1, 2),
            "memory_masks": memory_masks.to(torch.bool),
            "num_vertices": num_vertices,
            "vertex_labels": vertex_labels
        }

    def prepare_features_v2(
            self,
            pace_graphs_batch: List[ig.Graph],
            fixed_memory_len: Optional[int] = None,
    ):
        device = next(self.parameters()).device

        batch_size = len(pace_graphs_batch)

        # Initialize feature tensors directly on the target device
        vertex_label_features = torch.zeros(
            batch_size,
            self.max_num_vertices,
            self.vertex_label_cardinality,
            device=device,
        )
        vertex_position_features = torch.zeros(
            batch_size,
            self.max_num_vertices,
            self.max_num_vertices,
            device=device,
        )
        adjacency_matrices = torch.zeros(
            batch_size,
            self.max_num_vertices,
            self.max_num_vertices,
            device=device,
        )

        # Initialize masks
        total_heads = batch_size * self.num_heads
        source_masks = torch.ones(
            total_heads,
            self.max_num_vertices - 1,
            self.max_num_vertices - 1,
            device=device,
        )
        target_masks = torch.ones(
            batch_size * self.num_heads,
            self.max_num_vertices,
            self.max_num_vertices,
            device=device,
        )
        memory_masks = torch.ones(
            batch_size * self.num_heads,
            self.max_num_vertices,
            self.max_num_vertices - 1,
            device=device,
        )

        head_offset = 0

        for i, graph in enumerate(pace_graphs_batch):
            vertex_labels_i = graph.vs[self.graph_label_key]
            vertex_positions_i = graph.vs[self.graph_position_key]

            num_vertices_i = len(vertex_labels_i)

            # Handle padding
            pad_length = self.max_num_vertices - num_vertices_i
            if pad_length > 0:
                vertex_labels_i += [self.graph_label_output] * pad_length
                vertex_positions_i += [max(vertex_positions_i) + 1] * pad_length

            # One-hot encoding
            vertex_label_features[i, :, :] = F.one_hot(
                torch.tensor(vertex_labels_i, dtype=torch.long, device=device),
                num_classes=self.vertex_label_cardinality,
            )
            vertex_position_features[i, :, :] = F.one_hot(
                torch.tensor(vertex_positions_i, dtype=torch.long, device=device),
                num_classes=self.max_num_vertices,
            )

            # Adjacency matrix
            adj = torch.tensor(
                graph.get_adjacency().data,
                dtype=torch.float,
                device=device,
            )
            adjacency_matrices[i, :num_vertices_i, :num_vertices_i] = adj

            source_masks[head_offset:head_offset + self.num_heads, :num_vertices_i - 1,
            :num_vertices_i - 1] = torch.stack(
                [self.generate_mask(adj, num_vertices_i)[1:, 1:]] * self.num_heads,
                dim=0,
            )

            target_masks[head_offset:head_offset + self.num_heads, :num_vertices_i, :num_vertices_i] = torch.stack(
                [self.generate_mask(adj, num_vertices_i)] * self.num_heads,
                dim=0,
            )
            target_masks[head_offset:head_offset + self.num_heads, num_vertices_i:, num_vertices_i:] = torch.zeros(
                self.num_heads,
                self.max_num_vertices - num_vertices_i,
                self.max_num_vertices - num_vertices_i,
                device=device,
            )

            memory_len = num_vertices_i - 1 if fixed_memory_len is None else fixed_memory_len

            memory_masks[head_offset:head_offset + self.num_heads, :num_vertices_i, :memory_len] = torch.zeros(
                self.num_heads,
                num_vertices_i,
                memory_len,
                device=device,
            )
            memory_masks[head_offset:head_offset + self.num_heads, num_vertices_i:, -1:] = torch.zeros(
                self.num_heads,
                self.max_num_vertices - num_vertices_i,
                1,
                device=device,
            )

            head_offset += self.num_heads

        # We skip the start node for teacher forcing
        vertex_labels = [graph.vs[self.graph_label_key][1:] for graph in pace_graphs_batch]

        # Number of nodes
        num_vertices = [graph.vcount() for graph in pace_graphs_batch]

        return {
            "vertex_label_features": vertex_label_features,
            "vertex_position_features": vertex_position_features,
            "adjacency_matrices": adjacency_matrices,
            "source_masks": source_masks.to(torch.bool),
            "target_masks": target_masks.to(torch.bool).transpose(1, 2),
            "memory_masks": memory_masks.to(torch.bool),
            "num_vertices": num_vertices,
            "vertex_labels": vertex_labels
        }

    def encode_direct(self, features: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device

        vertex_label_features = features["vertex_label_features"].to(device)
        vertex_position_features = features["vertex_position_features"].to(device)
        adjacency_matrices = features["adjacency_matrices"].to(device)
        target_masks = features["target_masks"].to(device)

        vertex_position_embeddings = self.vertex_position_embed(vertex_position_features, adjacency_matrices)
        vertex_label_embeddings = self.vertex_label_embed(vertex_label_features)

        vertex_features = torch.cat(
            [
                vertex_label_embeddings,
                vertex_position_embeddings
            ],
            2
        )

        # here we set the source sequence and the tgt sequence for the teacher forcing
        # node 2 is the start symbol, shape: (bsiaze, max_n-1, nhid)
        src_inp = vertex_features.transpose(0, 1)

        # memory = self.encoder(src_inp,mask=src_mask)
        memory = self.encoder(src_inp, mask=target_masks)
        # shape ( batch_size, self.max_n-1, nhid): each batch, the first num_node - 1 rows are the representation of input nodes.
        memory = memory.transpose(0, 1).reshape(-1, self.max_num_vertices * self.ff_hidden_size)

        return self.fc1(memory), self.fc2(memory)

    def encode(self, labeled_graphs_batch: List[ig.Graph]) -> Tuple[torch.Tensor, torch.Tensor]:

        features = self.prepare_features(labeled_graphs_batch)

        return self.encode_direct(features)

    def reparameterize(
            self,
            mu: torch.Tensor,
            log_var: torch.Tensor,
            epsilon_scale: float = 0.01
    ) -> torch.Tensor:
        """
        Performs the reparameterization trick to sample from N(mu, std) during training.
        During evaluation, returns the mean value mu.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std) * epsilon_scale
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> List[ig.Graph]:
        """
        This is a sequence to sequence prediction model.
        Input: a graph (sequence of nodes)
        from a graph_label_input node, we use the transformer to predict the type of the next node
        and this process is continued until the graph_label_output node (or iterations reaches max_n)
        """
        device = next(self.parameters()).device

        batch_size = len(z)
        memory = self.fc3(z).reshape(-1, self.max_num_vertices, self.ff_hidden_size).transpose(0, 1)

        pace_graphs_batch = [ig.Graph(directed=True) for _ in range(batch_size)]
        for g in pace_graphs_batch:
            g.add_vertex()
            g.vs[0][self.graph_label_key] = self.graph_label_start
            g.vs[0][self.graph_position_key] = 0

            g.add_vertex()
            g.vs[1][self.graph_label_key] = self.graph_label_input
            g.vs[1][self.graph_position_key] = 1

        # memory = self.encoder(src_inp,mask=src_mask)

        finished = [False] * batch_size
        # the first two type of nodes are certain
        for idx in range(2, self.max_num_vertices):
            features = self.prepare_features_v2(pace_graphs_batch, self.max_num_vertices - 1)

            vertex_label_features = features["vertex_label_features"].to(device)
            vertex_position_features = features["vertex_position_features"].to(device)
            adjacency_matrices = features["adjacency_matrices"].to(device)
            target_masks = features["target_masks"].to(device)
            memory_masks = features["memory_masks"].to(device)

            pos_feat = self.vertex_position_embed(vertex_position_features, adjacency_matrices)
            node_feat = self.vertex_label_embed(vertex_label_features)
            node_feat = torch.cat([node_feat, pos_feat], 2)
            tgt_inp = node_feat.transpose(0, 1)

            out = self.decoder(tgt_inp, memory, tgt_mask=target_masks, memory_mask=memory_masks)
            out = out.transpose(0, 1)  # shape ( batch_size, self.max_n, nvrt)
            next_node_hidden = out[:, idx - 1, :]
            # add nodes
            type_scores = self.add_node(next_node_hidden)
            type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
            new_types = [np.random.choice(range(self.vertex_label_cardinality), p=type_probs[i]) for i in
                         range(len(pace_graphs_batch))]
            # add edges
            # just from the cneter node to the target node
            edge_scores = torch.cat([torch.stack([next_node_hidden] * (idx - 1), 1), out[:, :idx - 1, :]], -1)
            edge_scores = self.sigmoid(self.add_edge(edge_scores))

            for i, g in enumerate(pace_graphs_batch):
                if not finished[i]:
                    if idx < self.max_num_vertices - 1:
                        g.add_vertex(type=new_types[i])
                    else:
                        g.add_vertex(type=self.graph_label_output)
            for vi in range(idx - 2, -1, -1):
                ei_score = edge_scores[:, vi]  # 0 point to node 1
                random_score = torch.rand_like(ei_score)
                decisions = random_score < ei_score
                for i, g in enumerate(pace_graphs_batch):
                    if finished[i]:
                        continue
                    if new_types[i] == self.graph_label_output:
                        # if new node is graph_label_output, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0)
                                            if v.index != g.vcount() - 1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount() - 1)
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi + 1, g.vcount() - 1)

            for pace_graph in pace_graphs_batch:
                pace_graph.vs[self.graph_position_key] = self.compute_graph_positions(pace_graph)

        labeled_graphs_batch = [self.from_pace_graph_to_labeled_graph(pace_graph) for pace_graph in
                                pace_graphs_batch]

        return labeled_graphs_batch

    def loss_log_likelihood(
            self,
            num_vertices,
            vertex_labels,
            adjacency_matrices,
            decoder_output,
    ):
        device = next(self.parameters()).device

        batch_size = len(num_vertices)

        # Node predictions
        node_scores = self.add_node(decoder_output)  # Shape: [batch_size, max_num_vertices, num_node_classes]
        node_log_probs = self.logsoftmax2(node_scores)  # Shape: [batch_size, max_num_vertices, num_node_classes]

        # Initialize log likelihood
        log_likelihood = torch.tensor(0.0, device=device)

        # Create a tensor of shape (batch_size, max_length)
        padded_true_node_types = torch.zeros(batch_size, self.max_num_vertices, dtype=torch.long, device=device)

        # Concatenate and pad all vertex_labels at once
        padded_vertex_labels = [torch.tensor(vl[:self.max_num_vertices], dtype=torch.long) for vl in vertex_labels]
        padded_vertex_labels = torch.nn.utils.rnn.pad_sequence(padded_vertex_labels, batch_first=True, padding_value=0)

        padded_true_node_types[:, :padded_vertex_labels.shape[1]] = padded_vertex_labels.to(device)

        # Vectorized computation of node log likelihood
        graph_sizes_tensor = torch.tensor(num_vertices, device=device)
        valid_node_mask = torch.arange(self.max_num_vertices, device=device).expand(batch_size, self.max_num_vertices) < (graph_sizes_tensor - 1).unsqueeze(1)

        indices = padded_true_node_types.unsqueeze(2)  # Shape: [batch_size, max_num_vertices, 1]

        true_log_probs = torch.gather(node_log_probs, 2, indices).squeeze(2)  # Shape: [batch_size, max_num_vertices]

        masked_true_log_probs = true_log_probs * valid_node_mask.float()  # Shape: [batch_size, max_num_vertices]

        log_likelihood += masked_true_log_probs.sum()

        for i in range(batch_size):

            num_node_i = num_vertices[i] - 1
            num_potential_edges = int(num_node_i * (num_node_i - 1) / 2.0)
            edge_scores = torch.zeros(num_potential_edges, 2 * self.ff_hidden_size, device=device)  # Shape: [num_potential_edges, 2 * nhid]
            ground_truth = torch.zeros(num_potential_edges, 1, device=device)  # Shape: [num_potential_edges, 1]
            count = 0

            for idx in range(num_node_i - 1, 0, -1):
                repeated_output = decoder_output[i, idx].unsqueeze(0).repeat(idx, 1)  # Shape: [idx, nhid]
                connecting_output = decoder_output[i, :idx]  # Shape: [idx, nhid]
                edge_scores[count:count + idx, :] = torch.cat([repeated_output, connecting_output], dim=1)
                ground_truth[count:count + idx, :] = adjacency_matrices[i, 1:idx + 1, idx + 1].view(idx, 1)
                count += idx

            # Compute edge log-likelihood
            edge_log_likelihood = - F.binary_cross_entropy_with_logits(self.add_edge(edge_scores), ground_truth, reduction='sum')
            log_likelihood += edge_log_likelihood

        return log_likelihood

    def loss_log_likelihood_vectorized(
            self,
            num_vertices,
            vertex_labels,
            adjacency_matrices,
            decoder_output,
    ):
        device = next(self.parameters()).device

        batch_size = len(num_vertices)

        # Node predictions
        node_scores = self.add_node(decoder_output)  # Shape: [batch_size, max_num_vertices, num_node_classes]
        node_log_probs = self.logsoftmax2(node_scores)  # Shape: [batch_size, max_num_vertices, num_node_classes]

        # Initialize log likelihood
        log_likelihood = torch.tensor(0.0, device=device)

        # Create a tensor of shape (batch_size, max_length)
        padded_true_node_types = torch.zeros(batch_size, self.max_num_vertices, dtype=torch.long, device=device)

        # Concatenate and pad all vertex_labels at once
        padded_vertex_labels = [torch.tensor(vl[:self.max_num_vertices], dtype=torch.long) for vl in vertex_labels]
        padded_vertex_labels = torch.nn.utils.rnn.pad_sequence(padded_vertex_labels, batch_first=True, padding_value=0)

        padded_true_node_types[:, :padded_vertex_labels.shape[1]] = padded_vertex_labels.to(device)

        # Vectorized computation of node log likelihood
        graph_sizes_tensor = torch.tensor(num_vertices, device=device)
        valid_node_mask = torch.arange(self.max_num_vertices, device=device).expand(batch_size,
                                                                                    self.max_num_vertices) < (
                                  graph_sizes_tensor - 1).unsqueeze(1)

        indices = padded_true_node_types.unsqueeze(2)  # Shape: [batch_size, max_num_vertices, 1]

        true_log_probs = torch.gather(node_log_probs, 2, indices).squeeze(2)  # Shape: [batch_size, max_num_vertices]

        masked_true_log_probs = true_log_probs * valid_node_mask.float()  # Shape: [batch_size, max_num_vertices]

        log_likelihood += masked_true_log_probs.sum()

        for i in range(batch_size):
            num_node_i = num_vertices[i] - 1

            # Generate all possible pairs of node indices where idx_i > idx_j
            indices = torch.arange(num_node_i, device=device)
            idx_i, idx_j = torch.meshgrid(indices, indices, indexing='ij')
            mask = idx_i > idx_j
            edge_indices_i = idx_i[mask]
            edge_indices_j = idx_j[mask]

            # Retrieve the corresponding decoder outputs
            repeated_output = decoder_output[i, edge_indices_i]
            connecting_output = decoder_output[i, edge_indices_j]

            # Concatenate the outputs and compute edge scores
            edge_inputs = torch.cat([repeated_output, connecting_output], dim=1)
            edge_scores = self.add_edge(edge_inputs)

            # Retrieve ground truth from adjacency matrices
            ground_truth = adjacency_matrices[i, edge_indices_j + 1, edge_indices_i + 1].view(-1, 1)

            # Compute edge log-likelihood
            edge_log_likelihood = - F.binary_cross_entropy_with_logits(edge_scores, ground_truth, reduction='sum')
            log_likelihood += edge_log_likelihood

        return log_likelihood


    def loss_log_likelihood_full_vectorized(
            self,
            num_vertices,
            vertex_labels,
            adjacency_matrices,
            decoder_output,
    ):
        device = next(self.parameters()).device

        batch_size = len(num_vertices)

        # Node predictions
        node_scores = self.add_node(decoder_output)  # Shape: [batch_size, max_num_vertices, num_node_classes]
        node_log_probs = self.logsoftmax2(node_scores)  # Shape: [batch_size, max_num_vertices, num_node_classes]

        # Initialize log likelihood
        log_likelihood = torch.tensor(0.0, device=device)

        # Create a tensor of shape (batch_size, max_length)
        padded_true_node_types = torch.zeros(batch_size, self.max_num_vertices, dtype=torch.long, device=device)

        # Concatenate and pad all vertex_labels at once
        padded_vertex_labels = [torch.tensor(vl[:self.max_num_vertices], dtype=torch.long) for vl in vertex_labels]
        padded_vertex_labels = torch.nn.utils.rnn.pad_sequence(padded_vertex_labels, batch_first=True, padding_value=0)

        padded_true_node_types[:, :padded_vertex_labels.shape[1]] = padded_vertex_labels.to(device)

        # Vectorized computation of node log likelihood
        graph_sizes_tensor = torch.tensor(num_vertices, device=device)
        valid_node_mask = torch.arange(self.max_num_vertices, device=device).expand(batch_size,
                                                                                    self.max_num_vertices) < (
                                      graph_sizes_tensor - 1).unsqueeze(1)

        indices = padded_true_node_types.unsqueeze(2)  # Shape: [batch_size, max_num_vertices, 1]

        true_log_probs = torch.gather(node_log_probs, 2, indices).squeeze(2)  # Shape: [batch_size, max_num_vertices]

        masked_true_log_probs = true_log_probs * valid_node_mask.float()  # Shape: [batch_size, max_num_vertices]

        log_likelihood += masked_true_log_probs.sum()

        # Get maximum number of nodes minus one across the batch
        max_num_node = graph_sizes_tensor.max().item() - 1

        # Generate indices for nodes up to the maximum number of nodes
        indices = torch.arange(max_num_node, device=device)

        # Create meshgrid of indices for node pairs
        idx_i, idx_j = torch.meshgrid(indices, indices, indexing='ij')

        # Expand idx_i and idx_j to include the batch dimension
        idx_i = idx_i.unsqueeze(0).expand(batch_size, -1, -1)
        idx_j = idx_j.unsqueeze(0).expand(batch_size, -1, -1)

        # Create batch indices for each edge
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, max_num_node, max_num_node)

        # Create masks to select valid node pairs where idx_i > idx_j
        mask_upper = idx_i > idx_j

        # Create masks to ensure indices are within the valid range for each graph
        valid_i = idx_i < (graph_sizes_tensor[:, None, None] - 1)
        valid_j = idx_j < (graph_sizes_tensor[:, None, None] - 1)
        mask_valid = valid_i & valid_j

        # Combine masks
        total_mask = mask_upper & mask_valid

        # Apply masks to get valid indices
        batch_indices = batch_indices[total_mask]
        edge_indices_i = idx_i[total_mask]
        edge_indices_j = idx_j[total_mask]

        # Retrieve the corresponding decoder outputs
        repeated_output = decoder_output[batch_indices, edge_indices_i]
        connecting_output = decoder_output[batch_indices, edge_indices_j]

        # Concatenate the outputs and compute edge scores
        edge_inputs = torch.cat([repeated_output, connecting_output], dim=1)
        edge_scores = self.add_edge(edge_inputs)

        # Retrieve ground truth from adjacency matrices
        ground_truth = adjacency_matrices[batch_indices, edge_indices_j + 1, edge_indices_i + 1].view(-1, 1)

        # Compute edge log-likelihood
        edge_log_likelihood = -F.binary_cross_entropy_with_logits(
            edge_scores, ground_truth, reduction='sum'
        )

        # Accumulate log_likelihood
        log_likelihood += edge_log_likelihood

        return log_likelihood

    def loss_direct(
            self,
            features: Dict,
            beta: float = 0.005
    ):
        device = next(self.parameters()).device

        vertex_label_features = features["vertex_label_features"].to(device)
        vertex_position_features = features["vertex_position_features"].to(device)
        adjacency_matrices = features["adjacency_matrices"].to(device)  # adjacency_matrices: [batch_size, max_num_vertices, max_num_vertices]
        target_masks = features["target_masks"].to(device)
        memory_masks = features["memory_masks"].to(device)
        num_vertices = features["num_vertices"]
        vertex_labels = features["vertex_labels"]

        batch_size = len(num_vertices)

        mu, log_var = self.encode_direct(features)

        # Reparameterization trick
        latent_vectors = self.reparameterize(mu, log_var)

        # Transform latent vector for decoder
        decoder_states = self.fc3(latent_vectors).reshape(-1, self.max_num_vertices, self.ff_hidden_size).transpose(0, 1)

        # Embeddings
        vertex_position_embeddings = self.vertex_position_embed(vertex_position_features, adjacency_matrices)
        vertex_label_embeddings = self.vertex_label_embed(vertex_label_features)

        vertex_features = torch.cat(
            [
                vertex_label_embeddings,
                vertex_position_embeddings,
            ],
            dim=2,
        )

        # Decoder input
        decoder_input = vertex_features.transpose(0, 1)

        # Decode
        decoder_output = self.decoder(
            decoder_input,
            decoder_states,
            tgt_mask=target_masks,
            memory_mask=memory_masks
        ).transpose(0, 1)  # Shape: [batch_size, max_num_vertices, nhid]

        log_likelihood = self.loss_log_likelihood_full_vectorized(
            num_vertices,
            vertex_labels,
            adjacency_matrices,
            decoder_output,
        )

        # KL Divergence
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        total_loss = -log_likelihood + beta * kl_divergence

        return total_loss, -log_likelihood, kl_divergence


    def loss(
            self,
            labeled_graphs_batch: List[ig.Graph],
            beta: float = 0.005
    ):

        features = self.prepare_features(labeled_graphs_batch)

        return self.loss_direct(features, beta)
