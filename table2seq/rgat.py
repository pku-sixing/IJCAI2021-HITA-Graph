import torch
import torch.nn as nn
from utils.network.transformer.transformer_encoder import sequence_mask


class RGATLayer(nn.Module):

    def __init__(self, input_size, hidden_size, relation_nums, relation_matrix_num, learnable_weights, batch_first=True):
        super(RGATLayer, self).__init__()

        if batch_first is False:
            raise NotImplementedError()
        # currently support input_size == hidden_size
        if input_size != hidden_size:
            raise NotImplementedError()

        self.relation_matrix_num = relation_matrix_num
        self.hidden_size = hidden_size
        self.query_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.key_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.relational_weights = learnable_weights
        self.query_weights = nn.ModuleList(
            [nn.Embedding(num, hidden_size)
             for num in relation_nums])
        self.key_weights = nn.ModuleList(
            [nn.Embedding(num, hidden_size)
             for num in relation_nums])
        self.value_weights = nn.ModuleList(
            [nn.Embedding(num, hidden_size)
             for num in relation_nums])
        self.relation_embeddings = nn.ModuleList(
            [nn.Embedding(num, hidden_size)
             for num in relation_nums])

    def forward(self, query, key, value, src_len, relation_index_matrices):
        # => (batch, seq_len, input_size)
        batch_size = query.size()[0]
        seq_len = query.size()[1]
        input_size = query.size()[2]
        mask = ~sequence_mask(src_len)
        mask = mask.unsqueeze(1).expand(-1, seq_len, -1)

        if not self.relational_weights:
            # => (batch, seq_len, hidden_size)
            attention_score = torch.tanh(self.query_projection(query) + self.key_projection(key))
            attention_score = attention_score.view(batch_size, 1, seq_len, self.hidden_size)
            # => (batch, seq_len, seq_len, hidden_size)
            attention_score = attention_score.expand(-1, seq_len, -1, -1)
            # => (batch, seq_len, hidden_size)
            value = self.value_projection(value)

        output = None
        for i, relation_index_matrix in enumerate(relation_index_matrices):
            relation_index_matrix = relation_index_matrix.transpose(1, 2)
            # relation projection
            # => (batch, seq_len, seq_len, hidden_size)
            relation_matrix = self.relation_embeddings[i](relation_index_matrix)

            if self.relational_weights:
                query_weight = self.query_weights[i](relation_index_matrix)
                key_weight = self.key_weights[i](relation_index_matrix)
                value_weight = self.value_weights[i](relation_index_matrix)

                query = query.view(batch_size, 1, seq_len, self.hidden_size).expand(-1, seq_len, -1, -1)
                key = key.view(batch_size, 1, seq_len, self.hidden_size).expand(-1, seq_len, -1, -1)
                value = value.view(batch_size, 1, seq_len, self.hidden_size).expand(-1, seq_len, -1, -1)
                value = value_weight * value
                # => (batch, seq_len, seq_len, hidden_size)
                attention_score = torch.tanh(query_weight * query + (key_weight * key).transpose(1, 2))
            # (batch, seq_len, seq_len, hidden_size) * (batch, seq_len, seq_len, hidden_size)
            # => (batch, seq_len, seq_len, hidden_size)
            rel_att_score = torch.mul(relation_matrix, attention_score)
            # => (batch, seq_len, seq_len)
            rel_att_score = torch.sum(rel_att_score, -1)

            # => (batch, seq_len, seq_len)
            rel_att_score = rel_att_score.masked_fill(mask, -1e18)
            rel_att = torch.softmax(rel_att_score, -1)

            # => (batch, seq_len, hidden_size)
            if output is None:
                if self.relational_weights:
                    output = torch.sum((rel_att.unsqueeze(-1) * value), 2)
                else:
                    output = torch.matmul(rel_att, value)
            else:
                if self.relational_weights:
                    output += torch.sum((rel_att.unsqueeze(-1) * value), 2)
                else:
                    output += torch.matmul(rel_att, value)

        return output

if __name__ == '__main__':
    encoder = RGATLayer(20, 10, [10, 20], 2, True)

    word = torch.rand([7, 9, 20])
    src_len = torch.tensor([8, 6, 5, 7, 9, 5, 2], dtype=torch.long)
    relation_matrices = [(torch.rand(7, 9, 9) * 10).long(),
                           (torch.rand(7, 9, 9) * 20).long()]

    res = encoder(word, word, word, src_len, relation_matrices)
    print(res)