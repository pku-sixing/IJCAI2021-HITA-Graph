import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from table2seq.field_encoder import InfoboxTableEncoder
from utils.network.transformer.multi_head_attention import MultiHeadedAttention
from utils.network.transformer.transformer_encoder import sequence_mask


class HierarchicalIntraFieldTableEncoder(InfoboxTableEncoder):

    def __init__(self, args, src_embed, concat=False, dropout=0.5):
        super(HierarchicalIntraFieldTableEncoder, self).__init__(args, src_embed)
        self.concat = concat
        self.heads = args.hierarchical_field_encoder_heads
        self.field_size = args.field_key_embed_size
        self.word_size = self.field_word_embed_size
        self.hidden_size = args.hidden_size
        if concat:
            self.input_size = self.word_size + self.field_size
        else:
            self.input_size = self.word_size
        assert args.infobox_memory_bank_format == 'fwk_fwv_fk'
        self.input_projection = nn.Linear(self.input_size, self.hidden_size, bias=False)
        # self.word_context = torch.nn.Parameter(torch.Tensor(self.hidden_size).float(), requires_grad=True)
        self.word_context = nn.Linear(self.hidden_size, 1, bias=False)
        self.word_encoder = MultiHeadedAttention(self.heads, self.hidden_size)
        self.key_memory_projection = nn.Linear(self.hidden_size + self.field_key_embed_size + self.field_all_pos_embed_size, self.hidden_size, bias=False)
        self.attention_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_projection = nn.Linear(self.hidden_size * 2 + self.field_key_embed_size + self.field_all_pos_embed_size, self.hidden_size, bias=False)

    def forward(self, field_key, field_word, field_pos, field_tag, field_len, batch):
        """

        :param field_key: (seq_len, batch, input_size)
        :param field_word:  (seq_len, batch, input_size)
        :param field_pos: (batch)
        :param field_tag: (seq_len, batch)
        :param field_len: (seq_len, batch)
        :param batch
        :return:
        """
        # word-level encoder

        if self.concat:
            input = torch.cat([field_word, field_key], -1).transpose(0, 1)
        else:
            input = field_word.transpose(0, 1)
        # => (batch, seq_len, hidden_size)
        input_embed = self.input_projection(input)
        batch_size, seq_len, hidden_size = input_embed.size()

        # attribute-level encoder
        kv_pos = batch.attribute_kv_pos[0]
        # => (seq_len, batch)
        field_len_mask = sequence_mask(field_len).transpose(0, 1)
        # => (seq_len, batch) : attribute结束位置为1，其余为0
        field_mask = torch.mul((batch.attribute_word_local_bw_pos[0] <= 1), field_len_mask)
        field_kv_num = torch.sum(field_mask.float(), 0)
        max_kv_num = torch.max(field_kv_num).int().item()
        attribute_len_matrix = (field_kv_num - 1).unsqueeze(0).expand(seq_len, -1)
        # => (seq_len, batch):计算key与value对应位置
        kv_pos = kv_pos + (kv_pos == 0).long() * attribute_len_matrix.long()
        kv_pos[0, :] = 0

        kv_pos_masked = kv_pos * torch.where(field_len_mask == False, torch.tensor(-1, device=field_len_mask.device),
                                             torch.tensor(1, device=field_len_mask.device))
        # => (batch, max_kv_num, seq_len)
        kv_pos_masked = kv_pos_masked.transpose(0, 1).unsqueeze(1).expand(-1, max_kv_num, -1)

        select_index = torch.arange(max_kv_num, device=field_len_mask.device).unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, seq_len)
        # => (batch, max_kv_num, seq_len) 找到attribute与seq对应位置
        select_matrix = (select_index == kv_pos_masked)

        # MultiHeadAttention
        # => (batch * max_kv_num)
        field_value_len = torch.flatten(torch.sum(select_matrix, -1))
        # => (field_num = batch * valid_kv_num)
        field_value_len = field_value_len[field_value_len.nonzero()].squeeze()
        intra_field = torch.masked_select(input_embed, field_len_mask.transpose(0, 1).unsqueeze(-1))
        intra_field = torch.stack(torch.split(intra_field, self.hidden_size))
        intra_field = torch.split(intra_field, field_value_len.int().tolist())
        # => (field_num, max_field_value_len, hidden_size)
        intra_field = pad_sequence(intra_field, batch_first=True)

        # => (filed_num, max_field_value_len, hidden_size)
        mask = ~sequence_mask(field_value_len).unsqueeze(1)
        output_word, _ = self.word_encoder(intra_field, intra_field, intra_field, mask=mask)


        # => (batch, seq_len, hidden_size)  (key)
        output_word_att = torch.tanh(self.attention_projection(output_word))

        # Intra-field Attention
        # (key * query)
        # => (filed_num, max_field_value_len, 1)
        output_word_score = self.word_context(output_word_att)

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
        # => (batch, seq_len, hidden_size)
        output_field = torch.gather(summary_field, 1, kv_pos.transpose(0, 1).unsqueeze(-1).expand(-1, -1, hidden_size))
        output_word = torch.masked_select(output_word, ~mask.transpose(1, 2))
        output_word = torch.stack(torch.split(output_word, self.hidden_size))
        output_word = torch.split(output_word, field_len.int().tolist())
        output_word = pad_sequence(output_word, batch_first=True)


        # => (seq_len, batch, hidden_size * 2 + field_key_embed_size + field_all_pos_embed_size)
        output = torch.cat((output_word.transpose(0, 1), output_field.transpose(0, 1), field_key, field_pos), -1)
        # => (seq_len, batch, hidden_size)
        output = self.output_projection(output)
        field_key_memory_bank = torch.cat((output_field, field_key.transpose(0, 1), field_pos.transpose(0, 1)), -1)
        field_key_memory_bank = self.key_memory_projection(field_key_memory_bank)

        # get last output
        last_output_masks = (field_len - 1).unsqueeze(0).unsqueeze(2).expand(seq_len, -1, hidden_size)
        last_output = output.gather(0, last_output_masks)[0]
        return output, last_output, (output_word, output, field_key_memory_bank)

    def _encode(self, field_key, field_word, field_pos, field_tag, field_len, batch, context_node=None, relation_matrices=None):
        field_value_embed_output, field_summary, (field_word_key_memory_bank, field_word_val_memory_bank,
            field_key_memory_bank) = self.forward(field_key, field_word, field_pos, field_tag, field_len, batch)

        field_word_val_memory_bank = field_word_val_memory_bank.transpose(0, 1)
        table_memory_bank = (field_word_key_memory_bank, field_word_val_memory_bank, field_key_memory_bank)

        return field_value_embed_output, field_summary, table_memory_bank

