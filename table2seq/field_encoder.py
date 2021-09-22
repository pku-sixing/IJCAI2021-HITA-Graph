import torch
import torch.nn as nn
from table2seq import char_encoders
from seq2seq import rnn_helper
from utils.logger import logger
from utils import model_helper
VALID_FIELD_POS_INPUTS = {'local_pos_fw', 'local_pos_bw', 'field_kv_pos', 'field_kw_pos', 'none'}


class InfoboxTableEncoder(nn.Module):

    def __init__(self, args, src_embed):
        super(InfoboxTableEncoder, self).__init__()
        self.infobox_mode = 'standard'
        self.src_embed = src_embed
        self.field_key_embed_size = args.field_key_embed_size
        self.field_all_pos_embed_size = 0
        self.field_input_tags = set(args.field_input_tags.split(','))
        assert len(self.field_input_tags - VALID_FIELD_POS_INPUTS) == 0

        self.enabled_char_encoders = set(args.char_encoders.split(','))

        # Word embedding for representing infobox words
        self.field_key_embed = rnn_helper.build_embedding(args.field_vocab_size, args.field_key_embed_size)
        if 'field_key' in self.enabled_char_encoders:

            self.sub_field_key_char_encoder = char_encoders.T2STokenEncoder(args.char_encoder_type,
                                                                            dropout=args.dropout,
                                                                            vocab_size=args.sub_field_vocab_size,
                                                                            word_embed_size=args.field_key_embed_size,
                                                                            embed_size=args.field_key_embed_size, )
        else:
            self.sub_field_key_char_encoder = None

        if 'field_word' in self.enabled_char_encoders:
            self.sub_field_word_char_encoder = char_encoders.T2STokenEncoder(args.char_encoder_type,
                                                                             dropout=args.dropout,
                                                                             vocab_size=args.sub_field_word_vocab_size,
                                                                             word_embed_size=args.embed_size,
                                                                             embed_size=args.embed_size)
        else:
            self.sub_field_word_embed = None
            self.sub_field_word_char_encoder = None

        # Local Positions
        if 'local_pos_fw' in self.field_input_tags:
            self.local_pos_fw_embed = rnn_helper.build_embedding(args.max_field_intra_word_num,
                                                                 args.field_position_embedding_size)
            self.field_all_pos_embed_size += args.field_position_embedding_size
        else:
            self.local_pos_fw_embed = None

        if 'local_pos_bw' in self.field_input_tags:
            self.local_pos_bw_embed = rnn_helper.build_embedding(args.max_field_intra_word_num,
                                                                 args.field_position_embedding_size)
            self.field_all_pos_embed_size += args.field_position_embedding_size
        else:
            self.local_pos_bw_embed = None

        # Global Positions
        if 'field_kv_pos' in self.field_input_tags:
            self.field_kv_pos_embed = rnn_helper.build_embedding(args.max_kv_pairs_num,
                                                                 args.field_position_embedding_size)
            self.field_all_pos_embed_size += args.field_position_embedding_size
        else:
            self.field_kv_pos_embed = None

        if 'field_kw_pos' in self.field_input_tags:
            self.field_kw_pos_embed = rnn_helper.build_embedding(args.max_kw_pairs_num,
                                                                 args.field_position_embedding_size)
            self.field_all_pos_embed_size += args.field_position_embedding_size
        else:
            self.field_kw_pos_embed = None

        # POS Tag 的Embedding
        if args.field_tag_usage == 'general':
            self.field_tag_embed = rnn_helper.build_embedding(args.field_pos_tag_vocab_size,
                                                              args.field_tag_embedding_size)
            self.field_tag_embed_size = args.field_tag_embedding_size
        else:
            self.field_tag_embed = None
            self.field_tag_embed_size = 0

        # 对于Filed Word是否同时使用Field Word和标准Word的Embedding
        self.dual_field_word_embedding = args.dual_field_word_embedding
        if args.field_word_vocab_path != 'none':
            logger.info('[Model] Use a separate field word embedding ')
            self.field_word_embedding = rnn_helper.build_embedding(args.field_word_vocab_size, args.embed_size)
            self.field_word_embed_size = args.embed_size
            if args.dual_field_word_embedding:
                self.field_word_embed_size = args.embed_size * 2
        else:
            assert args.dual_field_word_embedding is False, 'requires a separate field vocab'
            self.field_word_embedding = self.src_embed
            self.field_word_embed_size = args.embed_size

        self.dual_attn = args.dual_attn != 'none'
        if self.dual_attn:
            self.dual_attn_projection = nn.Linear(self.field_key_embed_size + self.field_all_pos_embed_size +
                                                  self.field_tag_embed_size,
                                                  args.hidden_size, bias=False)

    def _represent_infobox(self, batch):
        if self.sub_field_key_char_encoder is None:
            field_key = self.field_key_embed(batch.attribute_key[0])
        else:
            field_key = self.field_key_embed(batch.attribute_key[0])
            field_key = self.sub_field_key_char_encoder.word_char_fusion(field_key,
                                                                         batch.sub_attribute_key, self.field_key_embed)
        if self.dual_field_word_embedding:
            if self.sub_field_word_char_encoder is None:
                field_word = self.field_word_embedding(batch.attribute_word[0])
            else:
                field_word = self.field_word_embedding(batch.attribute_word[0])
                field_word = self.sub_field_word_char_encoder.word_char_fusion(field_word,
                                                                               batch.sub_attribute_word,
                                                                               self.field_word_embedding)

            # field_word = self.field_word_embedding(batch.attribute_word[0])
            field_word_word = self.src_embed(batch.attribute_uni_word[0])
            field_word = torch.cat([field_word, field_word_word], -1)
        else:
            if self.sub_field_word_char_encoder is None:
                field_word = self.field_word_embedding(batch.attribute_word[0])
            else:
                field_word = self.field_word_embedding(batch.attribute_word[0])
                field_word = self.sub_field_word_char_encoder.word_char_fusion(field_word,
                                                                               batch.sub_attribute_word,
                                                                               self.field_word_embedding)

        field_pos = []
        if self.local_pos_fw_embed is not None:
            field_pos_st = self.local_pos_fw_embed(batch.attribute_word_local_fw_pos[0])
            field_pos.append(field_pos_st)
        if self.local_pos_bw_embed is not None:
            field_pos_ed = self.local_pos_bw_embed(batch.attribute_word_local_bw_pos[0])
            field_pos.append(field_pos_ed)
        if self.field_kv_pos_embed is not None:
            field_kv_pos = self.field_kv_pos_embed(batch.attribute_kv_pos[0])
            field_pos.append(field_kv_pos)
        if self.field_kw_pos_embed is not None:
            field_kw_pos = self.field_kw_pos_embed(batch.attribute_kw_pos[0])
            field_pos.append(field_kw_pos)
        if len(field_pos) > 0:
            field_pos = torch.cat(field_pos, -1)
        else:
            field_pos = None

        if self.field_tag_embed is None:
            field_tag = None
        else:
            field_tag = self.field_tag_embed(batch.attribute_word_tag[0])

        field_len = batch.attribute_word[1]
        return field_key, field_word, field_pos, field_tag, field_len

    def _encode(self, field_key, field_word, field_pos, field_tag, field_len, batch,
                context_node=None, relation_matrices=None):
        if self.infobox_mode == 'graph':
            field_value_embed_output, field_summary = self.forward(field_key, field_word, field_pos, field_tag, field_len,
                                                               batch, context_node, relation_matrices)
        else:
            field_value_embed_output, field_summary = self.forward(field_key, field_word, field_pos, field_tag,
                                                                   field_len, batch)
        field_value_memory_bank = field_value_embed_output.transpose(0, 1)
        field_non_word = [field_key]
        if field_pos is not None:
            field_non_word.append(field_pos)
        if field_tag is not None:
            field_non_word.append(field_tag)
        field_non_word = torch.cat(field_non_word, -1)
        field_key_memory_bank = field_non_word.transpose(0, 1)
        if self.dual_attn:
            field_key_memory_bank = self.dual_attn_projection(field_key_memory_bank)
        table_memory_bank = (field_value_memory_bank, field_key_memory_bank)

        return field_value_embed_output, field_summary, table_memory_bank

    def encode_infobox(self, batch, context_node):
        # Represent Encoder
        field_key, field_word, field_pos, field_tag, field_len = self._represent_infobox(batch)
        _, batch_size = batch.attribute_word[0].size()
        # 根据位置识别出来的，默认+1，再加上末尾有一个特殊的符号再+1
        field_seq_len = batch.attribute_kv_pos[0].max().item()
        if self.infobox_mode == 'graph':
            relation_matrix = [batch.attribute_graph[0]]
        else:
            relation_matrix = None
        # Encoding Infobox
        field_value_embed_output, field_summary, table_memory_bank = \
            self._encode(field_key, field_word, field_pos, field_tag, field_len, batch, context_node, relation_matrix)
        field_representations = [field_word, field_key, field_value_embed_output]
        if field_pos is not None:
            field_representations.append(field_pos)
        if field_tag is not None:
            field_representations.append(field_tag)
        return field_representations, field_summary, table_memory_bank, field_len

    def get_field_equivalent_input_size(self):
        return self.field_word_embed_size + self.field_key_embed_size + self.field_all_pos_embed_size + self.field_tag_embed_size


class FieldTableEncoder(InfoboxTableEncoder):

    def __init__(self, args, src_embed):
        super(FieldTableEncoder, self).__init__(args, src_embed)
        field_size = self.field_key_embed_size + self.field_all_pos_embed_size + self.field_tag_embed_size
        self.hidden_size = args.hidden_size
        self.core_matrix = nn.Linear(self.hidden_size + self.field_word_embed_size, self.hidden_size * 4)
        self.field_gate = nn.Linear(field_size, self.hidden_size * 2)

    def step(self, word_embed, field_embed, hidden_state=None, cell_memory=None):
        """

        :param word_embed: (batch, input_size)
        :param field_embed:  (batch, dim)
        :param hidden_state:  (batch, hidden_size)
        :param cell_memory:  (batch, hidden_size)

        :return:
        """

        # (batch, input_size + hidden_size)
        step_input = torch.cat([word_embed, hidden_state], dim=-1)
        # => (batch, 4 * hidden_size)
        states = self.core_matrix(step_input)
        i_t = torch.sigmoid(states[:, 0: self.hidden_size])
        f_t = torch.sigmoid(states[:, self.hidden_size * 1: self.hidden_size * 2])
        o_t = torch.sigmoid(states[:, self.hidden_size * 2: self.hidden_size * 3])
        c_hat_t = torch.tanh(states[:, self.hidden_size * 3: self.hidden_size * 4])

        # => (batch, 2 * hidden_size)
        field_states = self.field_gate(field_embed)
        l_t = torch.sigmoid(field_states[:, 0: self.hidden_size])
        z_hat_t = torch.tanh(field_states[:, self.hidden_size * 1: self.hidden_size * 2])

        # cell_memory
        c_out_t = f_t * cell_memory + i_t * c_hat_t + l_t * z_hat_t
        h_t = o_t * torch.tanh(c_out_t)

        return h_t, c_out_t

    def forward(self, field_key, field_word, field_pos, field_tag, field_len, batch, cell_memory=None,
                hidden_state=None):
        max_len, batch_size, _ = field_word.shape
        field_non_word = [field_key]
        if field_pos is not None:
            field_non_word.append(field_pos)
        if field_tag is not None:
            field_non_word.append(field_tag)
        field_non_word = torch.cat(field_non_word, -1)
        if cell_memory is None or hidden_state is None:
            assert hidden_state is None and cell_memory is None
            cell_memory = torch.zeros([batch_size, self.hidden_size], dtype=field_word.dtype, device=field_word.device)
            hidden_state = torch.zeros([batch_size, self.hidden_size], dtype=field_word.dtype, device=field_word.device)
        outputs = torch.zeros([max_len, batch_size, self.hidden_size], device=field_word.device)

        for t in range(max_len):
            hidden_state, cell_memory = self.step(field_word[t], field_non_word[t], hidden_state, cell_memory)
            outputs[t] = hidden_state

        # (seq_len, batch_size, dim) => (batch_size * seq_len, dim)
        # outputs = outputs.permute(1, 0, 2).contiguous()
        flatten_outputs = outputs.view(-1, self.hidden_size)
        flatten_indices = (field_len - 1) * batch_size + torch.arange(0, batch_size, dtype=torch.long,
                                                                      device=field_len.device)
        last_output = flatten_outputs.index_select(dim=0, index=flatten_indices)

        return outputs, last_output

# if __name__ == '__main__':
#     encoder = FieldTableEncoder(3, 4, 5)
#     # (3,3)
#     word_seq = torch.rand([10,7,3])
#     field_seq =  torch.rand([10,7,4])
#     src_len = torch.tensor([8,6,5,7,8,5,2], dtype=torch.long)
#
#     # res = encoder(word_seq, field_seq, src_len)
#     # print(res[-1])
#     res = encoder(word_seq[:,0:2,:], field_seq[:,0:2,:], src_len[0:2])
#     print(res[-1])
#     res = encoder(word_seq[:,0:1,:], field_seq[:,0:1,:], src_len[0:1])
#     print(res[-1])
#     res = encoder(word_seq[:, 1:2, :], field_seq[:, 1:2, :], src_len[1:2])
#     print(res[-1])
