import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from seq2seq.attention import GlobalAttention
from utils.network.transformer.multi_head_attention import MultiHeadedAttention
from utils.network.transformer.transformer_encoder import sequence_mask


class DotAttention(nn.Module):

    def __init__(self, input_size, learnable_query=True, batch_first=True):
        super(DotAttention, self).__init__()

        if batch_first is False:
            raise NotImplementedError()

        self.learnable_query = learnable_query
        if learnable_query:
            self.query = nn.Parameter(torch.Tensor(1, input_size).float(), requires_grad=True)

        self.attention = GlobalAttention(input_size)

    def forward(self, source, memory_bank, memory_lengths, no_concat=True):
        # => (seq_len, batch, input_size)
        batch_size = memory_bank.size()[0]
        seq_len = memory_bank.size()[1]

        if self.learnable_query:
            source = self.query.expand(batch_size, -1)
        output, _ = self.attention(source, memory_bank, memory_lengths=memory_lengths, no_concat=no_concat)
        output = torch.sum(output.transpose(0, 1), 1)

        return output


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, head_count, input_size, hidden_size, summary_position='first', batch_first=True):
        super(MultiHeadSelfAttention, self).__init__()

        if batch_first is False:
            raise NotImplementedError()

        self.summary_position = summary_position
        self.hidden_size = hidden_size
        self.output_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.attention = MultiHeadedAttention(head_count, input_size)

    def forward(self, source, memory_lengths):
        # => (seq_len, batch, input_size)
        batch_size = source.size()[0]

        mask = ~sequence_mask(memory_lengths).unsqueeze(1)
        output, _ = self.attention(source, source, source, mask=mask)
        output = self.output_projection(output)
        if self.summary_position == 'first':
            summary_output = output[:, 0, :]
        else:
            # get last output
            masks = (memory_lengths - 1).view(batch_size, 1, 1).expand(batch_size, 1, self.hidden_size)
            summary_output = output.gather(1, masks).squeeze()

        return output, summary_output

class HierarchicalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attn_type="mlp", batch_first=True):
        super(HierarchicalAttention, self).__init__()

        if batch_first is False:
            raise NotImplementedError()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn_type = attn_type
        if self.attn_type == "mlp":
            self.field_projection = nn.Linear(input_size * 2, 1, bias=False)
            self.intra_field_projection = nn.Linear(input_size * 3, 1, bias=False)
        elif self.attn_type == "general":
            self.field_projection = nn.Linear(input_size, hidden_size, bias=False)
            self.intra_field_projection = nn.Linear(input_size * 2, hidden_size, bias=False)
        self.value_projection = nn.Linear(input_size * 2, hidden_size, bias=False)

    def forward(self, copy_attn_query, field_memory_bank):
        """
        Args:
          context (FloatTensor): context hidden state on (t - 1) step ``(batch, dim`)`
          v_kv (FloatTensor): last layer of field_nodes ``(batch, max_kv_num, dim)``
          f_kwc (FloatTensor): intra-field-level representations ``(batch, seq_len, dim)``
           field_kv_pos (FloatTensor): kv position of each batch ``(batch, seq_len)`
          field_kv_num (FloatTensor): kv numbers of each batch ``(batch)`
          field_kw_len (FloatTensor): kw lens of each kv ``(batch, max_kv_num)`
          field_len_mask (FloatTensor): length of each kv ``(batch, seq_len)`
        Returns:
          FloatTensor: field_value
            ``(batch, dim)``
        """
        context = copy_attn_query
        v_kv = field_memory_bank[0]
        f_kwc = field_memory_bank[1]
        # (batch, seq_len) 标识每一个Field Value对应的地方
        kv_pos = field_memory_bank[2]
        field_kv_num = field_memory_bank[3]
        field_len_mask = field_memory_bank[4]
        select_matrix = field_memory_bank[5]

        batch_size = v_kv.size()[0]
        kv_len = v_kv.size()[1]
        input_size = v_kv.size()[2]
        seq_len = f_kwc.size()[1]

        if self.attn_type == "mlp":
            context_kv = context.view(batch_size, 1, input_size).expand(-1, kv_len, -1)
            kv_att_score = self.field_projection(torch.cat((v_kv, context_kv), -1)).squeeze()
            kv_mask = ~sequence_mask(field_kv_num)
            kv_att_score = kv_att_score.masked_fill(kv_mask, -float('inf'))
            kv_att = torch.softmax(kv_att_score, -1)
            # align kv_att to (batch seq hidden)
            kv_att = torch.gather(kv_att, 1, kv_pos)
            # align v_kv to (batch seq hidden)
            kv_kwc_align = torch.gather(v_kv, 1, kv_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size))

            # intra field attention
            context_kw = context.view(batch_size, 1, input_size).expand(-1, seq_len, -1)
            # => (batch, seq_len)
            kw_att_score = self.intra_field_projection(torch.cat((kv_kwc_align, context_kw, f_kwc), -1)).exp().squeeze()
            # => (batch, seq_len, 1)
            padding_masked_kv_pos = torch.where(field_len_mask, kv_pos, torch.ones_like(kv_pos) * -1).unsqueeze(-1)
            kv_pos_equal_matrix = padding_masked_kv_pos == padding_masked_kv_pos.permute(0, 2, 1)
            kv_pos_equal_matrix = kv_pos_equal_matrix.to(torch.float)
            kw_att_score_normalization = torch.matmul(kv_pos_equal_matrix, kw_att_score.unsqueeze(-1)).squeeze()
            kw_att_probs = kw_att_score / kw_att_score_normalization

            kw_att = kw_att_probs
            # value
            intra_field_value = self.value_projection(torch.cat((kv_kwc_align, f_kwc), -1))
            fusion_att = kv_att * kw_att
            fusion_att = torch.where(field_len_mask, fusion_att, torch.zeros_like(fusion_att))

            field_value = torch.matmul(fusion_att.unsqueeze(1), intra_field_value)
        elif self.attn_type == "general":
            # align v_kv to (batch seq hidden)
            kv_kwc_align = torch.gather(v_kv, 1, kv_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size))
            kv_att_score = self.field_projection(context).view(batch_size, 1, input_size)
            kv_att_score = torch.matmul(kv_att_score, kv_kwc_align.transpose(1, 2))
            kv_att_score = kv_att_score.masked_fill(~field_len_mask.unsqueeze(1), -float('inf'))
            kv_att = torch.softmax(kv_att_score, -1)

            # intra field attention
            context_kv = context.view(batch_size, 1, input_size).expand(-1, kv_len, -1)
            # => (batch, seq_len)
            kw_att_score = self.intra_field_projection(torch.cat((v_kv, context_kv), -1))
            kw_att_score = torch.matmul(kw_att_score, f_kwc.transpose(1, 2))
            kw_att_score = torch.where(select_matrix.bool(), kw_att_score, torch.ones_like(kw_att_score) * -float('inf'))
            kw_att = torch.softmax(kw_att_score, -1)
            mask = ~sequence_mask(field_kv_num)
            kw_att = kw_att.masked_fill(mask.unsqueeze(-1), 0)
            kw_att = torch.sum(kw_att, 1).unsqueeze(1)

            # value
            intra_field_value = self.value_projection(torch.cat((kv_kwc_align, f_kwc), -1))
            fusion_att = kv_att * kw_att

            field_value = torch.matmul(fusion_att, intra_field_value)
            fusion_att = fusion_att.squeeze()

        return field_value.permute(1, 0, 2), fusion_att

# if __name__ == '__main__':
#     encoder = HierarchicalAttention(20, 10)
#
#     query = torch.rand([7, 5, 20])
#     key = torch.rand([24, 9, 20])
#     context = torch.rand([7, 20])
#     field_kv_num = torch.tensor([3, 4, 3, 5, 5, 2, 2], dtype=torch.long)
#     field_kw_len = torch.tensor([8, 9, 5, 7, 8, 5, 2, 9, 5, 7, 8, 5,
#                                  2, 9, 5, 7, 8, 5, 2, 9, 5, 7, 8, 5], dtype=torch.long)
#
#     res = encoder(query, key, context, field_kv_num, field_kw_len, kv_pos)
#     print(res)
