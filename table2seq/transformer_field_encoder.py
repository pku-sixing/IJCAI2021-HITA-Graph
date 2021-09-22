import torch
import torch.nn as nn
from seq2seq.rnn_helper import build_embedding
from utils.network.transformer.transformer_encoder import TransformerEncoder
from table2seq.field_encoder import InfoboxTableEncoder

class TransformerFieldTableEncoder(InfoboxTableEncoder):

    def __init__(self, args, src_embed):
        super(TransformerFieldTableEncoder, self).__init__(args, src_embed)
        hidden_size = args.hidden_size
        heads = args.transformer_field_encoder_heads
        layers = args.transformer_field_encoder_layers
        dropout = args.dropout
        field_size = self.field_word_embed_size + args.field_key_embed_size \
                     + self.field_all_pos_embed_size + self.field_tag_embed_size
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(field_size, hidden_size, bias=False)
        self.transformer_encoder = TransformerEncoder(num_layers=layers, d_model=hidden_size, heads=heads,
                                                      d_ff=hidden_size, dropout=dropout, attention_dropout=dropout)



    def forward(self, field_key, field_word, field_pos, field_tag, field_len, batch, cell_memory=None, hidden_state=None):
        # """
        #
        # :param word_seq: (seq_len, batch, input_size)
        # :param field_seq:  (seq_len, batch, input_size)
        # :param seq_len: (batch)
        # :param cell_memory: (batch, hidden)
        # :param hidden_state: (batch, hidden)
        # :return:
        # """
        input_embed = [field_key, field_word]
        if field_pos is not None:
            input_embed.append(field_pos)
        if field_tag is not None:
            input_embed.append(field_tag)
        input_embed = torch.cat(input_embed, -1)
        input_embed = torch.tanh(self.input_projection(input_embed))
        _, outputs, length, last_output = self.transformer_encoder(input_embed, field_len)
        return outputs, last_output


if __name__ == '__main__':
    encoder = TransformerFieldTableEncoder(20, 12, 32)
    # (3,3)
    word_seq = torch.rand([9,7,20])
    field_seq = torch.rand([9,7,12])
    src_len = torch.tensor([8,9,5,7,8,5,2], dtype=torch.long)

    # res = encoder(word_seq, field_seq, src_len)
    # print(res[-1])
    res = encoder(word_seq[:,0:2,:], field_seq[:,0:2,:], src_len[0:2])
    print(res[-1].shape)
    # res = encoder(word_seq[:,0:1,:], field_seq[:,0:1,:], src_len[0:1])
    # print(res[-1])
    # res = encoder(word_seq[:, 1:2, :], field_seq[:, 1:2, :], src_len[1:2])
    # print(res[-1])
    #








