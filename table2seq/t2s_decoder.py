import torch
from torch import nn
import torch.nn.functional as F
from seq2seq import rnn_helper
from seq2seq.attention import GlobalAttention
from table2seq.attention_modules import HierarchicalAttention


class TableAwareDecoder(nn.Module):

    def __init__(self, hparams, embed):
        super(TableAwareDecoder, self).__init__()

        self.hidden_size = hidden_size = hparams.hidden_size
        self.embed_size = embed_size = hparams.embed_size
        self.n_layers = n_layers = hparams.dec_layers
        self.dropout = dropout = hparams.dropout
        self.vocab_size = hparams.tgt_vocab_size
        self.vocab_size_with_offsets = hparams.tgt_vocab_size_with_offsets
        self.rnn_type = hparams.rnn_type
        self.embed = embed
        self.dropout = nn.Dropout(dropout)
        self.add_last_generated_token = hparams.add_last_generated_token
        self.bow_loss = hparams.bow_loss > 0.0
        self.hierarchical_attention = hparams.hierarchical_infobox_attention

        self.enable_field_attn = hparams.enable_field_attn
        self.enable_query_attn = hparams.enable_query_attn
        self.complex_attention_query = hparams.complex_attention_query
        self.update_decoder_with_global_node = hparams.update_decoder_with_global_node


        self.src_copy_coverage = False
        if hparams.copy_coverage > 0:
            self.src_copy_coverage = True

        # Build Mode:不启用RNN Encoder
        self.text2text_mode = False
        self.table2text_mode = False

        if hparams.task_mode == 'text2text':
            self.text2text_mode = True
        elif hparams.task_mode == 'table2text':
            self.table2text_mode = True

        if self.table2text_mode:
            assert hparams.dual_attn == 'general'
            assert hparams.enable_query_attn is False and hparams.enable_field_attn is True
            assert hparams.copy is False and self.text2text_mode is False
        elif self.text2text_mode:
            assert hparams.dual_attn == 'general'
            assert hparams.enable_field_attn is False and hparams.field_copy is False
            assert self.table2text_mode is False

        self.share_copy_attn = hparams.share_copy_attn
        self.share_field_copy_attn = hparams.share_field_copy_attn

        # Build Dual Attention
        self.dual_attn = hparams.dual_attn != 'none'
        self.dual_attn_mode = hparams.dual_attn

        # Build Complex Query Key
        # if self.complex_attention_query:
        #     # 输入长度 = Z_t, y_t-1, g_l
        #     if self.update_decoder_with_global_node:
        #         input_size = hidden_size * 2 + embed_size
        #     else:
        #         input_size = hidden_size + embed_size
        #     self.complex_attention_query_projection = torch.nn.Linear(input_size, hidden_size, bias=False)
        # else:
        #     self.complex_attention_query_projection = None

            # Build Attention
        attention_size = 0
        if self.enable_query_attn:
            attention_size += hparams.hidden_size
            self.attention = GlobalAttention(hidden_size, coverage=self.share_copy_attn and self.src_copy_coverage,
                                             attn_type=hparams.attn_type, attn_func=hparams.attn_func)
        else:
            self.attention = None

        # Build Field Attention
        if self.enable_field_attn:
            attention_size += hparams.hidden_size
            if self.hierarchical_attention is False:
                self.field_attention = GlobalAttention(hidden_size, attn_type=hparams.attn_type,
                                                   attn_func=hparams.attn_func, dual_attn=self.dual_attn)
            else:
                self.field_attention = HierarchicalAttention(hidden_size, hidden_size,
                                                                attn_type=hparams.hierarchical_infobox_attention_type)
        else:
            self.field_attention = None


        if hparams.dual_attn == "gate":
            self.dual_attn_fusion_gate = nn.Linear(attention_size + hidden_size * hparams.dec_layers, hidden_size)
            self.out_size = 2 * hidden_size
            self.rnn_input_size = hidden_size * 1 + embed_size
        elif hparams.dual_attn == "gate_fusion":
            self.dual_attn_fusion_gate = nn.Linear(attention_size + hidden_size * hparams.dec_layers, hidden_size * 2)
            self.dual_attn_fusion_context_inputs = nn.Linear(hidden_size, hidden_size)
            self.dual_attn_fusion_table_context_inputs = nn.Linear(attention_size * hparams.dec_layers, hidden_size)
            self.out_size = 2 * hidden_size
            self.rnn_input_size = hidden_size * 1 + embed_size
        elif hparams.dual_attn == "selector":
            self.dual_attn_fusion_gate = nn.Linear(attention_size + embed_size, 1)
            self.out_size = 2 * hidden_size
            self.rnn_input_size = hidden_size * 1 + embed_size
        elif hparams.dual_attn == "general":
            self.dual_attn_fusion_gate = None
            self.out_size = attention_size + hidden_size
            self.rnn_input_size = attention_size + embed_size
        elif hparams.dual_attn == "none":
            self.dual_attn_fusion_gate = None
            self.out_size = attention_size + hidden_size
            self.rnn_input_size = attention_size + embed_size
        else:
            raise NotImplemented()
        if self.add_last_generated_token:
            self.out_size += embed_size
        if self.update_decoder_with_global_node:
            self.out_size += hidden_size

        # Build Query-Copy
        self.mode_num = 1
        self.src_copy = hparams.copy
        if hparams.copy:
            self.max_copy_token_num = hparams.max_copy_token_num
            if self.share_copy_attn:
                self.src_copy_attention = self.attention
            else:
                self.src_copy_attention = GlobalAttention(hidden_size, coverage=self.src_copy_coverage,
                                                          attn_type=hparams.copy_attn_type,
                                                          attn_func=hparams.copy_attn_func)
            self.mode_num += 1

        else:
            self.src_copy_attention = None

        self.field_copy = hparams.field_copy
        if self.field_copy:
            self.max_kw_pairs_num = hparams.max_kw_pairs_num
            self.mode_num += 1
            if self.share_field_copy_attn:
                assert self.field_attention is not None
                self.field_copy_attention = self.field_attention
            else:
                if self.hierarchical_attention is False:
                    self.field_copy_attention = GlobalAttention(hidden_size,
                                                            attn_type=hparams.attn_type,
                                                            attn_func=hparams.attn_func,
                                                            dual_attn=self.dual_attn)
                else:
                    self.field_copy_attention = HierarchicalAttention(hidden_size, hidden_size,
                                                                attn_type=hparams.hierarchical_infobox_attention_type)
        else:
            self.field_copy_attention = None

        # Selector
        if self.mode_num > 1:
            if hparams.mode_selector == 'mlp':
                self.mode_selector = nn.Sequential(
                    nn.Linear(self.out_size, self.hidden_size, bias=True),
                    torch.nn.Tanh(),
                    nn.Linear(self.hidden_size, self.mode_num, bias=False),
                )
            else:
                self.mode_selector = nn.Linear(self.out_size, self.mode_num, bias=False)

        # Decoder
        rnn_fn = rnn_helper.rnn_factory(hparams.rnn_type)
        self.rnn = rnn_fn(self.rnn_input_size, hidden_size, n_layers, dropout=dropout)

        # Vocabulary Predictor
        if self.add_last_generated_token:
            self.out_l1 = nn.Linear(self.out_size, self.hidden_size * 4)
            self.out_l2 = nn.Linear(self.hidden_size * 4, hparams.tgt_vocab_size, bias=False)
        else:
            self.out = nn.Linear(self.out_size, hparams.tgt_vocab_size, bias=False)

        if self.bow_loss:
            self.out_bow_l1 = nn.Linear(self.hidden_size * self.n_layers, self.hidden_size)
            self.out_bow_l2 = nn.Linear(self.hidden_size, hparams.tgt_vocab_size, bias=False)

    def forward(self, input, last_hidden, encoder_memory_bank, table_memory_bank, encoder_length, table_length,
                copied_token_equip_embedding=None, copy_attention_coverage_vector=None, global_node=None):
        """
        :param copy_attention_coverage_vector:
        :param copied_token_equip_embedding:
        :param input:[seq_len]
        :param last_hidden:
        :param encoder_memory_bank: (batch, seq_len, dim)
        :param encoder_length:
        :return:
        """
        # 倒数第二个一般是复合的Value
        field_value_memory_bank = table_memory_bank[-2]
        field_memory_bank = table_memory_bank if self.dual_attn else field_value_memory_bank

        if self.src_copy_attention is not None or self.field_copy_attention is not None:
            # Select copied embeddings
            original_word_embedding = self.embed(input)
            batch_size, equip_embedding_len, encoder_hidden_dim = copied_token_equip_embedding.shape
            # Index
            copied_select_index = input - self.vocab_size
            is_copied_tokens = (copied_select_index >= 0)
            zero_index = torch.zeros_like(copied_select_index, device=copied_select_index.device)
            valid_copied_select_index = torch.where(is_copied_tokens, copied_select_index, zero_index)

            # Offsets for flatten selection
            batch_offsets = torch.arange(0, batch_size * equip_embedding_len, equip_embedding_len, dtype=torch.long,
                                         device=copied_select_index.device)
            valid_copied_select_index = valid_copied_select_index + batch_offsets

            # Select
            flatten_equip_embedding = copied_token_equip_embedding.view(-1, encoder_hidden_dim)
            selected_equip_embedding = flatten_equip_embedding.index_select(index=valid_copied_select_index, dim=0)
            selected_equip_embedding = selected_equip_embedding.view(batch_size, encoder_hidden_dim)
            embedded = torch.where(is_copied_tokens.unsqueeze(-1), selected_equip_embedding, original_word_embedding)
            embedded = embedded.unsqueeze(0)  # (1,B,N)
        else:
            # Get the embedding of the current input word (last output word)
            embedded = self.embed(input).unsqueeze(0)  # (1,B,N)

        embedded = self.dropout(embedded)
        last_generated_token_embedded = embedded.squeeze(0)

        # Attention
        if self.rnn_type == 'lstm':
            attn_query = last_hidden[-1][0]
        else:
            attn_query = last_hidden[-1]

        # if self.complex_attention_query:
        #     complex_keys = [attn_query, last_generated_token_embedded]
        #     if self.update_decoder_with_global_node is not None:
        #         complex_keys.append(global_node)
        #     complex_keys = torch.cat(complex_keys, -1)
        #     attn_query = self.complex_attention_query_projection(complex_keys)
        attn_query = attn_query.unsqueeze(1)


        if self.enable_query_attn:
            context, attn_weights = self.attention(
                attn_query,
                encoder_memory_bank,
                coverage=copy_attention_coverage_vector if self.share_copy_attn and self.src_copy_coverage else None,
                memory_lengths=encoder_length,
                no_concat=True,
            )
        else:
            context, attn_weights = None, None

        if self.enable_field_attn:
            if self.hierarchical_attention is False:
                table_context, table_attn_weights = self.field_attention(
                    attn_query,
                    field_memory_bank,
                    memory_lengths=table_length,
                    no_concat=True,
                )
            else:
                table_context, table_attn_weights = self.field_attention(attn_query, field_memory_bank)
        else:
            table_context, table_attn_weights = None, None

        # Combine embedded input word and attended context, run through RNN
        if self.dual_attn_fusion_gate is None:
            rnn_input_list = [embedded]
            if self.enable_query_attn:
                assert context is not None
                rnn_input_list.append(context)
            if self.enable_field_attn:
                assert table_context is not None
                rnn_input_list.append(table_context)
            rnn_input = torch.cat(rnn_input_list, 2)
        else:
            dec_layer = last_hidden.shape[0]
            my_last_hidden = torch.cat([last_hidden[i:i + 1] for i in range(dec_layer)], 2)

            # Gate Input
            gate_input_list = [my_last_hidden]
            if self.enable_query_attn:
                assert context is not None
                gate_input_list.append(context)
            if self.enable_field_attn:
                assert table_context is not None
                gate_input_list.append(table_context)
            gate_input = torch.cat(gate_input_list, 2)

            if self.dual_attn_mode == 'gate_fusion':
                gate_outputs = self.dual_attn_fusion_gate(gate_input)
                context_input_gate = torch.sigmoid(gate_outputs[:, :, 0: self.hidden_size])
                table_context_input_gate = torch.sigmoid(gate_outputs[:, :, self.hidden_size * 1: self.hidden_size * 2])
                context_input = torch.tanh(self.dual_attn_fusion_context_inputs(context))
                table_context_input = torch.tanh(self.dual_attn_fusion_table_context_inputs
                                                 (torch.cat([my_last_hidden, table_context], 2)))
                dual_context = context_input_gate * context_input + table_context_input_gate * table_context_input
                rnn_input = torch.cat([embedded, dual_context], 2)
            else:
                dual_attn_gate = torch.sigmoid(self.dual_attn_fusion_gate(gate_input))
                dual_context = dual_attn_gate * context + (1.0 - dual_attn_gate) * table_context
                rnn_input = torch.cat([embedded, dual_context], 2)

        if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
            last_hidden = tuple([x.contiguous() for x in last_hidden])
        else:
            last_hidden = last_hidden.contiguous()

        # Run RNN
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)

        # Decoder上下文状态集合
        context_output_list = [output]
        if self.dual_attn_fusion_gate is None:
            if self.enable_query_attn:
                context_out = context.squeeze(0)
                context_output_list.append(context_out)
            if self.enable_field_attn:
                table_context_out = table_context.squeeze(0)
                context_output_list.append(table_context_out)
            if self.update_decoder_with_global_node:
                context_output_list.append(global_node)
        else:
            context_output_list.append(dual_context.squeeze(0))
        if self.add_last_generated_token:
            context_output_list.append(last_generated_token_embedded)
        context_output = torch.cat(context_output_list, 1)

        # Output Probs
        output_probs = []
        copy_attn_weights = None

        # Vocab Probs
        if self.add_last_generated_token:
            tmp = self.dropout(torch.tanh(self.out_l1(context_output)))
            output = self.out_l2(tmp)
        else:
            output = self.out(context_output)
        vocab_probs = F.softmax(output, dim=1)
        output_probs.append(vocab_probs)

        # Query Probs
        if self.src_copy:
            # Copy Attention
            if self.rnn_type == 'lstm':
                copy_attn_query = hidden[-1][0]
            else:
                copy_attn_query = hidden[-1]

            # if self.complex_attention_query:
            #     complex_keys = [copy_attn_query, last_generated_token_embedded]
            #     if self.update_decoder_with_global_node is not None:
            #         complex_keys.append(global_node)
            #     complex_keys = torch.cat(complex_keys, -1)
            #     copy_attn_query = self.complex_attention_query_projection(complex_keys)
            copy_attn_query = copy_attn_query.unsqueeze(1)

            copy_context, copy_attn_weights = self.src_copy_attention(
                copy_attn_query,
                encoder_memory_bank,
                coverage=copy_attention_coverage_vector,
                memory_lengths=encoder_length - 1,  # mask the last token <sos>
                no_concat=True,
                mask_first_token=True,
            )
            copy_probs = copy_attn_weights.squeeze(0)

            # padding copy
            assert copy_probs.shape[1] <= self.max_copy_token_num
            if self.max_copy_token_num - copy_probs.shape[1] > 0:
                padding_probs = torch.zeros(
                    [copy_probs.shape[0], self.max_copy_token_num - copy_probs.shape[1]],
                    device=copy_probs.device)
                src_copy_probs = torch.cat([copy_probs, padding_probs], -1)
                output_probs.append(src_copy_probs)
            else:
                output_probs.append(copy_probs)
        # Field Copy
        if self.field_copy:
            # Copy Attention
            if self.rnn_type == 'lstm':
                copy_attn_query = hidden[-1][0]
            else:
                copy_attn_query = hidden[-1]
            # if self.complex_attention_query:
            #     complex_keys = [copy_attn_query, last_generated_token_embedded]
            #     if self.update_decoder_with_global_node  is not None:
            #         complex_keys.append(global_node)
            #     complex_keys = torch.cat(complex_keys, -1)
            #     copy_attn_query = self.complex_attention_query_projection(complex_keys)
            #     # copy_attn_query = copy_attn_query.unsqueeze(1)


            if self.hierarchical_attention is False:
                table_context, table_attn_weights = self.field_copy_attention(
                    copy_attn_query,
                    field_memory_bank,
                    memory_lengths=table_length,
                    no_concat=True,
                )
            else:
                table_context, table_attn_weights = self.field_copy_attention(
                    copy_attn_query,
                    field_memory_bank,
                )
            table_probs = table_attn_weights.squeeze(0)

            # padding copy probs
            assert table_probs.shape[1] <= self.max_kw_pairs_num , table_probs.shape[1]
            if self.max_kw_pairs_num - table_probs.shape[1] > 0:
                padding_probs = torch.zeros(
                    [table_probs.shape[0], self.max_kw_pairs_num - table_probs.shape[1]],
                    device=table_probs.device)
                table_copy_probs = torch.cat([table_probs, padding_probs], -1)
                output_probs.append(table_copy_probs)
            else:
                output_probs.append(table_probs)

        # Mode Fusion
        assert self.mode_num == len(output_probs)
        if self.mode_num > 1:
            selector = torch.softmax(self.mode_selector(context_output), -1)
            for i in range(self.mode_num):
                probs = output_probs[i]
                select_src_probs = selector[:, i:i+1]
                output_probs[i] = output_probs[i] * select_src_probs
        output = torch.cat(output_probs, -1)
        output = torch.log(output + 1e-20)

        return output, hidden, attn_weights, copy_attn_weights
