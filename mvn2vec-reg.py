'''

@author: sezin
'''

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import gc


class Model_AllCombined(nn.Module):
    def __init__(self, num_net, len_common_nodes, embedding_dim, embed_freq, batch_size, negative_sampling_size=10):

        super(Model_AllCombined, self).__init__()
        self.n_embedding = len_common_nodes
        self.embed_freq = embed_freq
        self.num_net = num_net
        self.node_embeddings = nn.ModuleList()
        self.neigh_embeddings = nn.ModuleList()
        for n_net in range(self.num_net):  # len(G)
            self.node_embeddings.append(nn.Embedding(len_common_nodes, embedding_dim))
            self.neigh_embeddings.append(nn.Embedding(len_common_nodes, embedding_dim))

        self.negative_sampling_size = negative_sampling_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

    def forward(self, count, shuffle_indices_nets, nodesidx_nets, neighidx_nets, gamma):
        cost = []
        for i in range(self.num_net):

            batch_indices = shuffle_indices_nets[i][count:count + self.batch_size]

            nodesidx = torch.LongTensor(nodesidx_nets[i][batch_indices]).cuda()
            node_emb = self.node_embeddings[i](Variable(nodesidx)).view(len(batch_indices), -1).unsqueeze(
                2)

            sumr1_inner = 0
            for vr in range(self.num_net):
                sumr1_inner += self.node_embeddings[vr](Variable(nodesidx)).view(len(batch_indices), -1).unsqueeze(2)

            r1 = torch.dist(node_emb, sumr1_inner / self.num_net, p=2) ** 2

            neighsidx = torch.LongTensor(
                neighidx_nets[i][batch_indices]).cuda()  # neigbors of b_n, it is set, converted list
            neigh_emb = self.neigh_embeddings[i](Variable(neighsidx)).unsqueeze(2).view(len(batch_indices), -1,
                                                                                        self.embedding_dim)
            loss_positive = nn.functional.logsigmoid(
                torch.bmm(neigh_emb, node_emb)).squeeze().mean()  # only 1, converges to dot product

            sumr2_inner = 0
            for vr in range(self.num_net):
                sumr2_inner += self.neigh_embeddings[vr](Variable(neighsidx)).view(len(batch_indices), -1).unsqueeze(2)

            r2 = torch.dist(neigh_emb, sumr2_inner / self.num_net, p=2) ** 2

            ###############positive finished##########

            negative_context = self.embed_freq.multinomial(
                len(batch_indices) * neigh_emb.size(1) * self.negative_sampling_size,
                replacement=True).cuda()
            # higher freq index higher chance to be sampled
            negative_context_emb = self.neigh_embeddings[i](negative_context).view(len(batch_indices), -1,
                                                                                   self.embedding_dim).neg()

            loss_negative = nn.functional.logsigmoid(torch.bmm(negative_context_emb, node_emb)).squeeze().sum(1).mean(
                0)  # sum through all 10 negative words/columns, row size is batch size, takes average respct to batch size

            sumr2_inner_neg = 0
            for vr in range(self.num_net):
                sumr2_inner_neg += self.neigh_embeddings[vr](negative_context).view(len(batch_indices), -1).unsqueeze(2)
            r2_neg = torch.dist(neigh_emb, sumr2_inner_neg / self.num_net, p=2) ** 2

            cost.append(
                loss_positive + loss_negative - gamma * (r1 + r2)/self.batch_size - gamma * (r1 + r2_neg)/self.batch_size)  # loss is updated with gamma

            torch.cuda.empty_cache()
            gc.collect()
        return -sum(cost) / len(cost)

    def input_embeddings(self):
        embeddings = []
        for n_net in range(self.num_net):  # len(G)
            embeddings.append(self.node_embeddings[n_net].weight.data)
        return embeddings

    def input_embeddings_dict(self):
        embeddings = []
        for n_net in range(self.num_net):  # len(G)
            embeddings.append(self.node_embeddings[n_net])
        return embeddings

    def save_embedding(self, file_name, id2word):
        embeds = self.node_embeddings.weight.data
        fo = open(file_name, 'a+')
        for idx in range(len(embeds)):
            word = id2word[idx]
            embed = ' '.join(embeds[idx])
            fo.write(word + ' ' + embed + '\n')
        fo.close()