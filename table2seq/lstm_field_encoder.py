import torch
import torch.nn as nn
from seq2seq.rnn_helper import build_embedding
from table2seq.field_encoder import InfoboxTableEncoder


class LSTMFieldTableEncoder(InfoboxTableEncoder):

    def __init__(self, args, src_embed, concat=True, dropout=0.5):
        super(LSTMFieldTableEncoder, self).__init__(args, src_embed)
        self.concat = concat
        self.hidden_size = args.hidden_size
        self.field_size = args.field_key_embed_size
        self.input_size = self.field_word_embed_size
        assert args.field_input_tags == "none"
        if concat:
            self.word_encoder = LSTM(self.input_size + self.field_size, self.hidden_size)
        else:
            self.word_encoder = LSTM(self.input_size, self.hidden_size)
        assert args.infobox_memory_bank_format == 'fwk_fwv_fk'
        self.attribute_encoder = LSTM(self.hidden_size, self.hidden_size)
        self.positional_embedding = build_embedding(500, 5)
        self.output_projection = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

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
            input = torch.cat([field_word, field_key], -1)
        else:
            input = field_word
        output_word = self.word_encoder(input, field_len)   # (seq_len, batch, hidden_size)
        word_mask = sequence_mask(field_len).transpose(0, 1).unsqueeze(-1).expand(-1, -1, self.hidden_size)

        # attribute-level encoder
        attribute_mask = sequence_mask(field_len).transpose(0, 1)   # (seq_len, batch)
        attribute_mask = torch.mul((batch.attribute_word_local_bw_pos[0] <= 1), attribute_mask)  # (seq_len, batch) : attribute结束位置为1，其余为0
        attribute_len = torch.sum(attribute_mask.float(), 0)       # (batch) : 每个batch中, attribute的数量
        attribute_index = attribute_mask.unsqueeze(-1).expand(-1, -1, self.hidden_size)  # (seq_len, batch, hidden_size)

        attribute_input = torch.masked_select(output_word.transpose(0,1), attribute_index.transpose(0,1)) #根据attribute位置取出隐状态
        # attribute_input = torch.gather(output.transpose(0,1), 1, attribute_index.transpose(0,1))

        attribute_input = torch.stack(torch.split(attribute_input, self.hidden_size))
        attribute_input = torch.split(attribute_input, attribute_len.int().tolist())
        output_attribute = self.attribute_encoder(nn.utils.rnn.pad_sequence(attribute_input), attribute_len) # (attribute_len, batch, hidden_size)

        # 将attribute与word拼接起来
        kv_pos = batch.attribute_kv_pos[0]
        attribute_len_matrix = (attribute_len - 1).unsqueeze(0).expand(kv_pos.size()[0], -1)
        kv_pos = kv_pos + (kv_pos == 0).long() * attribute_len_matrix.long()  # (seq_len, batch):计算key与value对应位置
        kv_pos[0, :] = 0
        kv_pos = kv_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size)   # (seq_len, batch, hidden_size)

        output_attribute = torch.gather(output_attribute, 0, kv_pos) #(seq_len, batch, hidden_size) : attribute按word位置gather
        output_attribute = torch.mul(output_attribute, word_mask)

        output = torch.cat((output_word, output_attribute), -1) #(seq_len, batch, hidden_size * 2)
        output = self.output_projection(output)  #(seq_len, batch, hidden_size)

        # get last output
        masks = (field_len - 1).unsqueeze(0).unsqueeze(2).expand(output.size()[0], -1, self.hidden_size)
        last_output = output.gather(0, masks)[0]
        return output, last_output, (output_word, output, output_attribute)

    def _encode(self, field_key, field_word, field_pos, field_tag, field_len, batch, context_node=None, relation_matrices=None):
        field_value_embed_output, field_summary, (field_word_key_memory_bank, field_word_val_memory_bank,
            field_key_memory_bank) = self.forward(field_key, field_word, field_pos, field_tag, field_len, batch)
        field_word_key_memory_bank = field_word_key_memory_bank.transpose(0, 1)
        field_word_val_memory_bank = field_word_val_memory_bank.transpose(0, 1)
        field_key_memory_bank = field_key_memory_bank.transpose(0, 1)
        table_memory_bank = (field_word_key_memory_bank, field_word_val_memory_bank, field_key_memory_bank)

        return field_value_embed_output, field_summary, table_memory_bank


class LSTM(nn.Module):
    def __init__(self, input_size, lstm_hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=1)
        # self.in_dropout = nn.Dropout(config.dropout)
        # self.out_dropout = nn.Dropout(config.dropout)

    def forward(self, src, src_lengths):
        '''
        src: [batch_size, slen, input_size]
        src_lengths: [batch_size]
        '''

        self.lstm.flatten_parameters()

        # src = self.in_dropout(src)

        new_src_lengths, sort_index = torch.sort(src_lengths, dim=0, descending=True)
        new_src = torch.index_select(src, dim=1, index=sort_index)

        packed_src = nn.utils.rnn.pack_padded_sequence(new_src, new_src_lengths, enforce_sorted=True)
        packed_outputs, (src_h_t, src_c_t) = self.lstm(packed_src)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        unsort_index = torch.argsort(sort_index)
        outputs = torch.index_select(outputs, dim=1, index=unsort_index)

        # outputs = self.out_dropout(outputs)

        return outputs

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

# if __name__ == '__main__':
#     encoder = LSTMFieldTableEncoder(20, 12, 32)
#     # (3,3)
#     word_seq = torch.rand([9,7,20])
#     field_seq = torch.rand([9,7,12])
#     field_len = torch.tensor([[1,1,3,2,1,2,1,1,0],
#                               [2,1,1,1,3,2,1,2,1],
#                               [1,1,2,1,1,0,0,0,0]], dtype=torch.long).transpose(0,1)
#     kv_pos = torch.tensor([[1, 2, 3, 3, 3, 4, 4, 5, 0],
#                            [1, 1, 2, 3, 4, 4, 4, 5, 5],
#                            [1, 2, 3, 3, 4, 0, 0, 0, 0]], dtype=torch.long).transpose(0, 1)
#     src_len = torch.tensor([8,9,5,7,8,5,2], dtype=torch.long)
#
#     res = encoder(word_seq[:,0:3,:], field_seq[:,0:3,:], src_len[0:3], field_len, kv_pos)
#     print(res[-1].shape)








