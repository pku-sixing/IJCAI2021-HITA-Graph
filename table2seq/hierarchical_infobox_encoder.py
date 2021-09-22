import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from table2seq.field_encoder import InfoboxTableEncoder
from table2seq.rgat import RGATLayer
from utils.network.transformer.multi_head_attention import MultiHeadedAttention
from utils.network.transformer.transformer_encoder import sequence_mask


class HierarchicalInfoboxEncoder(InfoboxTableEncoder):

    def __init__(self, args, src_embed):
        super(HierarchicalInfoboxEncoder, self).__init__(args, src_embed)
        self.infobox_mode = 'graph'
        hidden_size = args.hidden_size
        self.rgat_layers = args.hierarchical_infobox_rgat_layers
        self.heads = args.hierarchical_infobox_encoder_heads
        self.hidden_size = hidden_size
        self.learnable_global_node = args.hierarchical_infobox_rgat_learnable_global_node
        if self.learnable_global_node:
            self.global_node = nn.Parameter(torch.randn(1, 1, hidden_size))
        dropout = args.dropout
        field_size = self.field_word_embed_size + args.field_key_embed_size \
                     + self.field_all_pos_embed_size + self.field_tag_embed_size
        self.input_projection = nn.Linear(field_size, hidden_size, bias=False)
        self.word_encoder = MultiHeadedAttention(self.heads, self.hidden_size)
        self.word_context = nn.Linear(self.hidden_size, 1, bias=False)
        self.hierarchical_attention = args.hierarchical_infobox_attention
        self.rgat = RGATLayer(hidden_size, hidden_size, [13, 1], 2, args.hierarchical_infobox_rgat_relational_weights)
        self.field_value_memory_projection = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)


    def forward(self, field_key, field_word, field_pos, field_tag, field_len, batch, context_node, relation_matrices):
        # """
        #
        # :param word_seq: (seq_len, batch, input_size)
        # :param field_seq:  (seq_len, batch, input_size)
        # :param seq_len: (batch)
        # :param cell_memory: (batch, hidden)
        # :param hidden_state: (batch, hidden)
        # :return:
        # """

        # 输入是已Embedding编码的 Field_key 和 Field_Word Sequence
        batch_size = field_key.size()[1]
        seq_len = field_key.size()[0]

        input_embed = [field_key, field_word]
        if field_pos is not None:
            input_embed.append(field_pos)
        if field_tag is not None:
            input_embed.append(field_tag)
        # => (batch, seq_len, field_size)
        input_embed = torch.cat(input_embed, -1).transpose(0, 1)
        # => (batch, seq_len, hidden_size)
        input_embed = torch.tanh(self.input_projection(input_embed))

        # attribute-level encoder
        kv_pos = batch.attribute_kv_pos[0]
        # => (seq_len, batch)
        field_len_mask = sequence_mask(field_len).transpose(0, 1)
        # => (seq_len, batch) : attribute结束位置为1，其余为0
        field_mask = torch.mul((batch.attribute_word_local_bw_pos[0] <= 1), field_len_mask)
        field_kv_num = torch.sum(field_mask.float(), 0)
        max_kv_num = torch.max(field_kv_num).int().item()
        field_len_matrix = (field_kv_num - 1).unsqueeze(0).expand(seq_len, -1)
        # => (seq_len, batch):计算key与value对应位置
        kv_pos = kv_pos + (kv_pos == 0).long() * field_len_matrix.long()
        kv_pos[0, :] = 0

        kv_pos_masked = kv_pos * torch.where(field_len_mask == False, torch.tensor(-1, device=field_len_mask.device),
                                             torch.tensor(1, device=field_len_mask.device))
        # => (batch, max_kv_num, seq_len)
        kv_pos_masked = kv_pos_masked.transpose(0, 1).unsqueeze(1).expand(-1, max_kv_num, -1)

        # => (batch, 1, seq_len)
        max_field_len = torch.max(field_len).item()
        select_index = torch.arange(max_kv_num, device=field_len_mask.device).unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, max_field_len)
        # => (batch, max_kv_num, seq_len) 找到attribute与seq对应位置
        select_matrix = (select_index == kv_pos_masked).float()

        # MultiHeadAttention
        # => (batch, seq)
        field_len_mask = sequence_mask(field_len)
        # => (batch * max_kv_num)
        field_value_len = torch.flatten(torch.sum(select_matrix, -1))
        # => (field_num = batch * valid_kv_num)
        field_value_len = field_value_len[field_value_len.nonzero()].squeeze()
        intra_field = torch.masked_select(input_embed, field_len_mask.unsqueeze(-1).expand(-1, -1, self.hidden_size))
        intra_field = torch.stack(torch.split(intra_field, self.hidden_size))
        intra_field = torch.split(intra_field, field_value_len.int().tolist())
        # => (field_num, max_field_value_len, hidden_size)
        intra_field = pad_sequence(intra_field, batch_first=True)


        # => (filed_num, max_field_value_len, hidden_size)
        mask = ~sequence_mask(field_value_len).unsqueeze(1)
        output_word, _ = self.word_encoder(intra_field, intra_field, intra_field, mask=mask)

        # Intra-field Attention
        # (key * query)
        # => (filed_num, max_field_value_len, 1)
        output_word_score = self.word_context(output_word)

        # => (field_num, max_field_value_len, 1)
        output_word_score = output_word_score.masked_fill(mask.transpose(1, 2), -1e18)
        # => (field_num, max_field_value_len, 1) 按attribute softmax计算attention值
        output_field_att = torch.softmax(output_word_score, 1)

        # attention * value
        # => (field_num, max_field_value_len, hidden_size)
        summary_field = output_field_att * output_word
        # => (field_num, hidden_size)
        summary_field = torch.sum(summary_field, 1)

        summary_field = torch.split(summary_field, field_kv_num.int().tolist())
        # => (batch, max_kv_num, hidden_size)
        summary_field = pad_sequence(summary_field, batch_first=True)

        # test input
        # context_node = torch.randn(batch_size, 1, self.hidden_size, device=field_key.device)
        # relation_matrices = [(torch.rand(batch_size, max_kv_num + 2, max_kv_num + 2, device=field_key.device) * 10).long(),
        #                      (torch.rand(batch_size, max_kv_num + 2, max_kv_num + 2, device=field_key.device) * 20).long()]

        graph = self.build_graph(context_node, summary_field)# => (batch, max_kv_num + 2, hidden_size)
        for _ in range(self.rgat_layers):
            graph = self.rgat(graph, graph, graph, (field_kv_num + 2), relation_matrices)

        global_node = graph[:, 0, :]
        # context_node = graph[:, 1, :]
        field_nodes = graph[:, 2:, :]

        # => (batch, seq_len, hidden_size)
        output_field = torch.gather(field_nodes, 1, kv_pos.transpose(0, 1).unsqueeze(-1).expand(-1, -1, self.hidden_size))

        output_word = torch.masked_select(output_word, ~mask.transpose(1, 2))
        output_word = torch.stack(torch.split(output_word, self.hidden_size))
        output_word = torch.split(output_word, field_len.int().tolist())
        # => (batch, seq_len, hidden_size)
        output_word = pad_sequence(output_word, batch_first=True)

        output = torch.cat((output_word.transpose(0, 1), output_field.transpose(0, 1)), -1)
        output = self.field_value_memory_projection(output)
        field_key_memory_bank = output_field

        if self.hierarchical_attention:
            # kv+kw, last_output, (kv, kw, kv_pos, kv_num, kw_lens, kv+kw, kv)
            return output, global_node, (field_nodes, output_word, kv_pos.transpose(0, 1), field_kv_num,
                                         field_len_mask, select_matrix, output, field_key_memory_bank)
        else:
            # kv+kw, last_output, (kv+kw, kv)
            return output, global_node, (output, field_key_memory_bank)



    def build_graph(self, context_node, field_nodes):
        batch_size = field_nodes.size()[0]
        if self.learnable_global_node:
            global_node = self.global_node.expand(batch_size, -1, -1)
        else:
            global_node = torch.randn(batch_size, 1, self.hidden_size, device=field_nodes.device)

        graph = torch.cat((global_node, context_node.unsqueeze(1), field_nodes), 1)
        return graph

    def _encode(self, field_key, field_word, field_pos, field_tag, field_len, batch,
                context_node=None, relation_matrices=None):
        if self.hierarchical_attention:
            field_value_embed_output, field_summary, (field_nodes, output_kwc, field_kv_pos, field_kv_num, field_len_mask,
                                                      select_matrix, field_value_memory_bank, field_key_memory_bank) = \
                self.forward(field_key, field_word, field_pos, field_tag, field_len, batch, context_node,
                             relation_matrices)
            field_value_memory_bank = field_value_memory_bank.transpose(0, 1)
            table_memory_bank = (field_nodes, output_kwc, field_kv_pos, field_kv_num, field_len_mask,
                                 select_matrix, field_value_memory_bank, field_key_memory_bank)
        else:
            field_value_embed_output, field_summary, (field_value_memory_bank, field_key_memory_bank) = \
                self.forward(field_key, field_word, field_pos, field_tag, field_len, batch, context_node,
                             relation_matrices)
            field_value_memory_bank = field_value_memory_bank.transpose(0, 1)
            table_memory_bank = (field_value_memory_bank, field_key_memory_bank)



        return field_value_embed_output, field_summary, table_memory_bank







