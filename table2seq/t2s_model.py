import torch
from torch import nn
from seq2seq import rnn_helper
from seq2seq.attention import GlobalAttention
from seq2seq.bridge import LinearBridge, AttentionBridge
from table2seq.hierarchical_field_encoder import HierarchicalFieldTableEncoder
from table2seq.hierarchical_infobox_encoder import HierarchicalInfoboxEncoder
from table2seq.hierarchical_intra_field_encoder import HierarchicalIntraFieldTableEncoder
from table2seq.lstm_field_encoder import LSTMFieldTableEncoder
from table2seq.t2s_decoder import TableAwareDecoder
from seq2seq.encoder import RNNEncoder
from utils.logger import logger
from utils import model_helper
from table2seq.field_encoder import FieldTableEncoder
from table2seq.transformer_field_encoder import TransformerFieldTableEncoder

class InfoSeq2Seq(nn.Module):
    def __init__(self, args, src_field, tgt_field):
        super(InfoSeq2Seq, self).__init__()

        logger.info('[MODEL] Preparing the InfoSeq2Seq model')
        self.drop_out = nn.Dropout(args.dropout)
        self.beam_length_penalize = args.beam_length_penalize
        self.hidden_size = args.hidden_size
        self.teach_force_rate = args.teach_force_rate
        # 如果打开 Table2Text模式，那么就只做Table2Text的生成,Text2Text 同理
        self.table2text_mode = False
        self.text2text_mode = False
        if args.task_mode == 'text2text':
            self.text2text_mode = True
        elif args.task_mode == 'table2text':
            self.table2text_mode = True

        self.enable_query_encoder = True
        self.enable_field_encoder = True

        self.enabled_char_encoders = set(args.char_encoders.split(','))

        if self.table2text_mode:
            self.enable_query_encoder = False
            assert args.copy is False
            assert args.bridge == 'none' or args.bridge == 'general'
        if self.text2text_mode:
            self.enable_field_encoder = False
            assert args.field_copy is False
            assert args.bridge == 'none' or args.bridge == 'general'

            # Word embedding for representing dialogue words
        self.embed_size = args.embed_size
        if args.share_embedding:
            self.src_embed = rnn_helper.build_embedding(args.tgt_vocab_size_with_offsets, args.embed_size)
            self.dec_embed = self.src_embed
            enc_embed = self.src_embed
            dec_embed = self.src_embed
        else:
            self.src_embed = rnn_helper.build_embedding(args.src_vocab_size, args.embed_size)
            self.dec_embed = rnn_helper.build_embedding(args.tgt_vocab_size_with_offsets, args.embed_size)
            dec_embed = self.dec_embed

        if 'src' in self.enabled_char_encoders:
            self.sub_src_embed = rnn_helper.build_embedding(args.sub_src_vocab_size, args.embed_size)

        # 是否使用Dialogue的Pos Tag
        if args.add_pos_tag_embedding and self.enable_query_encoder:
            self.add_pos_tag_embedding = True
            self.src_pos_tag_embed_size = args.field_tag_embedding_size
            self.src_pos_tag_embed = rnn_helper.build_embedding(args.src_tag_vocab_size,
                                                                self.src_pos_tag_embed_size)
        else:
            self.src_pos_tag_embed_size = 0
            self.add_pos_tag_embedding = False
            self.src_pos_tag_embed = None


        # Build Field Encoder
        if self.enable_field_encoder:
            if args.field_encoder == 'lstm':
                field_encoder = FieldTableEncoder(args=args, src_embed=self.src_embed)
            elif args.field_encoder == 'transformer':
                field_encoder = TransformerFieldTableEncoder(args=args, src_embed=self.src_embed)
            elif args.field_encoder == 'hierarchical_lstm':
                field_encoder = LSTMFieldTableEncoder(args=args, src_embed=self.src_embed)
            elif args.field_encoder == 'hierarchical_field':
                field_encoder = HierarchicalFieldTableEncoder(args=args, src_embed=self.src_embed)
            elif args.field_encoder == 'hierarchical_intra_field':
                field_encoder = HierarchicalIntraFieldTableEncoder(args=args, src_embed=self.src_embed)
            elif args.field_encoder == 'hierarchical_infobox':
                field_encoder = HierarchicalInfoboxEncoder(args=args, src_embed=self.src_embed)
            self.field_equivalent_input_size = field_encoder.get_field_equivalent_input_size()
            self.field_encoder = field_encoder
        else:
            self.field_equivalent_input_size = None
            self.field_encoder = None

        if self.enable_query_encoder:
            encoder = RNNEncoder(hparams=args, embed=enc_embed, tag_embed=self.src_pos_tag_embed)
            self.encoder = encoder
            if args.enc_birnn:
                self.birnn_down_scale = torch.nn.Linear(args.hidden_size*2, args.hidden_size, False)
            else:
                self.birnn_down_scale = None
                # Build Bridge
            self.bridge_mode = args.bridge
            if args.bridge == 'general' or args.bridge == 'fusion':
                bridge = LinearBridge(args.bridge, args.rnn_type, args.hidden_size, args.enc_layers,
                                      args.dropout, args.enc_birnn)
            elif args.bridge == 'attention' or args.bridge == 'field_attention' or args.bridge == 'fusion_attention':
                bridge = AttentionBridge(args.bridge, args.hidden_size, args.dropout, args.enc_birnn)
            elif args.bridge =='clue_attention' or args.bridge =='clue_attention2':
                bridge = AttentionBridge('attention', args.hidden_size, args.dropout, args.enc_birnn)
                self.posterior_bridge = AttentionBridge('attention', args.hidden_size, args.dropout, args.enc_birnn)
                if args.bridge == 'clue_attention' or args.bridge == 'clue_attention2':
                    self.clue_query_projection = torch.nn.Linear((int(args.enc_birnn) + 1) * args.hidden_size * 2,
                                                                 args.hidden_size * (int(args.enc_birnn) + 1))
            elif args.bridge == 'post_field_attention' or args.bridge == 'clue_field_attention' \
                    or args.bridge == 'clue_field_attention2':
                bridge = AttentionBridge('field_attention', args.hidden_size, args.dropout, args.enc_birnn)
                self.posterior_bridge = AttentionBridge('field_attention', args.hidden_size, args.dropout, args.enc_birnn)
                if args.bridge == 'clue_field_attention'  or args.bridge == 'clue_field_attention2':
                    self.clue_query_projection = torch.nn.Linear((int(args.enc_birnn) + 1) * args.hidden_size*2,
                                                                 args.hidden_size * (int(args.enc_birnn) + 1))
            elif args.bridge == 'post_attn_fusion':
                bridge = AttentionBridge('fusion_attention', args.hidden_size, args.dropout, args.enc_birnn)
                self.posterior_bridge = LinearBridge('general', args.rnn_type, args.hidden_size, args.enc_layers,
                                                     args.dropout)
            elif args.bridge == 'post_fusion' or args.bridge == 'post_fusion2':
                bridge = LinearBridge('fusion', args.rnn_type, args.hidden_size, args.enc_layers, args.dropout, args.enc_birnn)
                post_bridge = LinearBridge('general', args.rnn_type, args.hidden_size, args.enc_layers, args.dropout, args.enc_birnn)
                self.posterior_bridge = post_bridge
            elif args.bridge == 'none':
                bridge = None
            else:
                raise NotImplementedError()
            self.bridge = bridge
        else:
            self.encoder = None
            assert args.bridge == 'none', 'table2text mode does not support bridge'

        # Build Decoder
        decoder = TableAwareDecoder(hparams=args, embed=dec_embed)
        self.decoder = decoder

        # Build SRC-Copy Attention
        self.src_copy_mode = False
        self.copy_coverage = False
        if args.copy:
            assert self.table2text_mode is False
            self.src_copy_mode = True
            self.max_copy_token_num = args.max_copy_token_num
            # State-to-Input Projection:
            self.add_state_to_copy_token = args.add_state_to_copy_token
            copy_state_input_dim = args.embed_size
            if self.add_pos_tag_embedding:
                copy_state_input_dim += self.src_pos_tag_embed_size
            if self.add_state_to_copy_token:
                copy_state_input_dim += self.hidden_size
            if copy_state_input_dim > args.embed_size:
                self.copied_state_to_input_embed_projection = nn.Linear(copy_state_input_dim,
                                                                        args.embed_size, bias=False)
            else:
                self.copied_state_to_input_embed_projection = None
            # Coverage
            if args.copy_coverage > 0.0:
                self.copy_coverage = True

        # Build Field-SRC Copy 
        self.field_copy_mode = False
        if args.field_copy:
            self.field_copy_mode = True
            self.max_kw_pairs_num = args.max_kw_pairs_num
            # State-to-Input Projection
            self.field_state_to_input_embed_projection = nn.Linear(self.field_equivalent_input_size + self.hidden_size,
                                                                   args.embed_size, bias=False)

        # Special Tokens
        self.tgt_sos_idx = tgt_field.vocab.stoi['<sos>']
        self.tgt_eos_idx = tgt_field.vocab.stoi['<eos>']
        self.tgt_unk_idx = tgt_field.vocab.stoi['<unk>']
        self.tgt_pad_idx = tgt_field.vocab.stoi['<pad>']

    @classmethod
    def build_s2s_model(cls, args, src_field, tgt_field):
        seq2seq = cls(args, src_field, tgt_field)
        return seq2seq

    def beam_search_decoding(self, hparams, batch, beam_width):
        max_len = hparams.infer_max_len
        min_len = hparams.infer_min_len
        src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
        vocab_size = self.decoder.vocab_size_with_offsets
        raw_vocab_size = hparams.tgt_vocab_size
        copy_vocab_size = vocab_size - raw_vocab_size
        device = tgt.device

        disable_multiple_copy = hparams.disable_multiple_copy
        penalize_repeat_tokens_rate = hparams.penalize_repeat_tokens_rate

        with torch.no_grad():
            # Encoding Query
            query_summary, encoder_memory_bank = self.encode_query(batch)
            # Encode Infobox
            field_representations, field_summary, table_memory_bank, field_len = self.encode_infobox(batch,
                                                                                                     query_summary)
            # Bridge
            hidden, prior_hidden, mse_loss = self.bridge_init(batch, query_summary, field_summary,
                                                              encoder_memory=(encoder_memory_bank, src_len),
                                                              field_memory=(table_memory_bank[-2], field_len))
            # Generate Equivalent Embeddings And Prepare Embeddings
            equivalent_embeddings, copy_mask, coverage_losses, copy_attention_coverage_vector = \
                self.generate_equivalent_embeddings(batch, encoder_memory_bank, field_representations)

            # Multiply Infobox
            # field_value_memory_bank = model_helper.create_beam(table_memory_bank[0], beam_width, 0)
            # field_key_memory_bank = model_helper.create_beam(table_memory_bank[1], beam_width, 0)
            # table_memory_bank = (field_value_memory_bank, field_key_memory_bank)
            new_table_memory_bank = []
            for old in table_memory_bank:
                new_table_memory_bank.append(model_helper.create_beam(old, beam_width, 0))
            table_memory_bank = tuple(new_table_memory_bank)
            if field_summary is not None:
                field_summary = model_helper.create_beam(field_summary, beam_width, 0)

            # Multiply Query
            encoder_memory_bank = model_helper.create_beam(encoder_memory_bank, beam_width, 0)


            # Multiply Equivalent Embeddings
            if equivalent_embeddings is not None:
                equivalent_embeddings = model_helper.create_beam(equivalent_embeddings, beam_width, 0)
            if copy_attention_coverage_vector is not None:
                copy_attention_coverage_vector = model_helper.create_beam(copy_attention_coverage_vector, beam_width, 0)

            # Decoding
            if disable_multiple_copy:
                copied_indicator = torch.zeros([batch_size * beam_width, copy_vocab_size], device=device)

            repeat_indicator = torch.ones([batch_size * beam_width, vocab_size], device=device)
            output_probs = torch.zeros(max_len, batch_size * beam_width, vocab_size, device=device)
            last_token = torch.ones([batch_size * beam_width], dtype=torch.long, device=device) * self.tgt_sos_idx
            token_outputs = [last_token]
            decoded_seq_len = torch.zeros([batch_size * beam_width], device=device)
            ending_flags = torch.zeros([batch_size * beam_width], dtype=torch.bool, device=device)
            padding_label = torch.ones([batch_size * beam_width], dtype=torch.long, device=device) * self.tgt_pad_idx

            padding_logits = -1e20 * torch.ones([batch_size, beam_width, beam_width], device=device)
            padding_logits[:, :, 0] = 0.0

            diverse_decoding_offsets = torch.arange(0, beam_width, device=device) * hparams.diverse_decoding
            diverse_decoding_offsets = -diverse_decoding_offsets.repeat(batch_size * beam_width).view(batch_size,
                                                                                                      beam_width,
                                                                                                      beam_width)

            # multiply beam
            beam_scores = torch.zeros(max_len, batch_size, beam_width, device=device)
            init_beam_score = torch.zeros(batch_size, beam_width, device=device) + -1e20
            init_beam_score[:, 0] = 0.0
            beam_scores[0] = init_beam_score

            hidden = model_helper.create_beam(hidden, beam_width, 1)
            src_len = model_helper.create_beam(src_len, beam_width, 0)
            field_len = model_helper.create_beam(field_len, beam_width, 0)

            for t in range(1, max_len):
                prob_output, hidden, attn_weights, copy_attn_weights = self.decoder(
                    last_token, hidden, encoder_memory_bank, table_memory_bank, src_len, field_len,
                    copied_token_equip_embedding=equivalent_embeddings,
                    copy_attention_coverage_vector=copy_attention_coverage_vector,
                    global_node=field_summary,
                )

                # Ensure the min len
                if min_len != -1 and t <= min_len:
                    prob_output[:, self.tgt_eos_idx] = -1e20

                # Mask the unk
                if hparams.disable_unk_output:
                    prob_output[:, self.tgt_unk_idx] = -1e20

                prob_output = repeat_indicator * prob_output
                if disable_multiple_copy:
                    copy_probs = prob_output[:, raw_vocab_size:]
                    copy_probs += copied_indicator
                    prob_output[:, raw_vocab_size:] = copy_probs

                # Beam prob_output
                # (batch*beam, vocab_size)=> (batch, beam, vocab_size)
                prob_output = prob_output.view(batch_size, beam_width, vocab_size)

                # Select top-k for each beam => (batch, beam, beam]
                # topk_scores_in_each_beam 每个Beam中的Top-K，每个Beam中Top-K的Index（词）
                topk_scores_in_each_beam, topk_indices_in_each_beam = prob_output.topk(beam_width, dim=-1)

                # add diverse decoding penalize
                if t > 1:
                    topk_scores_in_each_beam = topk_scores_in_each_beam + diverse_decoding_offsets

                # avoid copying a finished beam(batch_size, beam)
                tmp_ending_flags = ending_flags.view(batch_size, beam_width, 1)
                finished_offsets = - topk_scores_in_each_beam * tmp_ending_flags
                finished_offsets = finished_offsets + padding_logits * tmp_ending_flags

                # compute total beam scores
                total_beam_scores = beam_scores.sum(dim=0)
                topk_total_scores_in_each_beam = topk_scores_in_each_beam + total_beam_scores.unsqueeze(-1)
                topk_total_scores_in_each_beam = topk_total_scores_in_each_beam + finished_offsets

                # decoded_seq_len (batch, beam_width, 1)
                tmp_decoded_seq_len = torch.where(ending_flags, decoded_seq_len, decoded_seq_len + 1)
                if self.beam_length_penalize == 'avg':
                    length_factor = tmp_decoded_seq_len.view(batch_size, beam_width, 1)
                    # 限制最大惩罚长度
                    length_factor = torch.min(length_factor, torch.ones_like(length_factor) * 20)
                    topk_total_scores_in_each_beam = topk_total_scores_in_each_beam / length_factor

                # Group beams/tokens to the corresponding group
                # => (batch, beams*beams)
                topkk_scores_in_batch_group = topk_total_scores_in_each_beam.view(batch_size, -1)
                topkk_tokens_in_batch_group = topk_indices_in_each_beam.view(batch_size, -1)

                # current step scores
                topk_current_scores_in_each_beam = topk_scores_in_each_beam + finished_offsets
                topk_current_scores_in_batch_group = topk_current_scores_in_each_beam.view(batch_size, -1)

                # Select top-K beams in each top-k*top-k batch group => (batch, beam)
                # topk_scores_in_batch 每个Batch Group中选择出来的K个分数，topk_indices_in_batch 每个Batch Group对应的 Beam Index
                topk_scores_in_batch, topk_indices_in_batch = topkk_scores_in_batch_group.topk(beam_width, -1)
                # => top_k tokens
                topk_tokens_in_batch = topkk_tokens_in_batch_group.gather(dim=-1, index=topk_indices_in_batch)

                # Current Scores (batch, beam*beam) => (batch, beam) 每个Batch内的Top-K分数
                current_scores_in_batch = topk_current_scores_in_batch_group.gather(dim=-1, index=topk_indices_in_batch)

                # => (batch, beam) the selected beam ids of the new top-k beams
                selected_last_beam_indices_in_batch = topk_indices_in_batch // beam_width

                # => this indices  is used to select batch*beam
                flatten_indices = selected_last_beam_indices_in_batch.view(-1)
                flatten_offsets = torch.arange(0, batch_size * beam_width, beam_width, device=device)
                flatten_offsets = flatten_offsets.repeat_interleave(beam_width, -1)
                flatten_offsets = flatten_offsets.view(-1)
                flatten_indices = flatten_indices + flatten_offsets

                # select hidden, from outputted hidden, select original beam
                # (layer, batch*beam, dim) => (layer, batch*beam, dim)
                if not isinstance(hidden, tuple) and not isinstance(hidden, list):
                    resorted_hidden = hidden.index_select(dim=1, index=flatten_offsets)
                    hidden = resorted_hidden
                else:
                    hidden = tuple([x.index_select(dim=1, index=flatten_offsets) for x in hidden])

                # select beam_scores, from original
                # (max_len, batch_size, beam_width) => (max_len, batch_size*beam_width)
                flatten_beam_scores = beam_scores.view(-1, batch_size * beam_width)
                resorted_beam_scores = flatten_beam_scores.index_select(dim=1, index=flatten_indices).view(
                    beam_scores.shape)
                beam_scores = resorted_beam_scores

                # select token_outputs, from original
                # (max_len, batch_size*beam_width)
                next_token_outputs = []
                for time_id in range(t):
                    current_tensor = token_outputs[time_id]
                    resorted_current_tensor = current_tensor.index_select(dim=0, index=flatten_indices)
                    next_token_outputs.append(resorted_current_tensor)
                token_outputs = next_token_outputs

                # select ending_flags (batch*beam), from original
                resorted_ending_flags = ending_flags.index_select(dim=0, index=flatten_indices)
                ending_flags = resorted_ending_flags

                # select copied_indicator (batch*beam), from original
                if disable_multiple_copy:
                    copied_indicator = copied_indicator.index_select(dim=0, index=flatten_indices)

                repeat_indicator = repeat_indicator.index_select(dim=0, index=flatten_indices)

                # selected sequence_length
                decoded_seq_len = tmp_decoded_seq_len.index_select(dim=0, index=flatten_indices)

                # select copy attention coverage
                if copy_attention_coverage_vector is not None:
                    # From original
                    resorted_copy_attention_coverage_vector = \
                        copy_attention_coverage_vector.index_select(dim=0, index=flatten_indices)
                    copy_attention_coverage_vector = resorted_copy_attention_coverage_vector
                    copy_attn_weights = copy_attn_weights.index_select(dim=1, index=flatten_indices)
                    copy_attention_coverage_vector = copy_attention_coverage_vector + copy_attn_weights.squeeze(0)

                # update beams
                last_token = torch.where(ending_flags, padding_label, topk_tokens_in_batch.view(-1))
                token_outputs.append(last_token)
                beam_scores[t] = current_scores_in_batch * (last_token.view(batch_size, beam_width) != self.tgt_pad_idx)

                # Is finished
                is_finished = last_token == self.tgt_eos_idx
                ending_flags |= is_finished
                if torch.sum(ending_flags) == batch_size * beam_width:
                    break

                # Adding mask
                if disable_multiple_copy:
                    new_copy_indicator = torch.zeros_like(copied_indicator, device=copied_indicator.device)
                    copy_index = last_token - raw_vocab_size
                    is_copied_token = copy_index >= 0
                    copy_index = torch.max(torch.zeros_like(copy_index, device=copy_index.device), copy_index)
                    new_copy_indicator = new_copy_indicator.scatter(-1, copy_index.unsqueeze(-1), -1e10)
                    copied_indicator = copied_indicator + new_copy_indicator

                # Repeat repeat_indicator
                new_repeat_indicator = torch.ones_like(repeat_indicator, device=repeat_indicator.device)
                new_repeat_indicator = new_repeat_indicator.scatter(-1, last_token.unsqueeze(-1),
                                                                    penalize_repeat_tokens_rate)
                repeat_indicator = repeat_indicator * new_repeat_indicator

            return output_probs, token_outputs, beam_scores.view(-1, batch_size * beam_width)

    def greedy_search_decoding(self, hparams, batch):
        # Set variables
        max_len = hparams.infer_max_len
        min_len = hparams.infer_min_len
        src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
        vocab_size = self.decoder.vocab_size_with_offsets
        device = tgt.device

        with torch.no_grad():

            # Encoding Query
            query_summary, encoder_memory_bank = self.encode_query(batch)
            # Encode Infobox
            field_representations, field_summary, table_memory_bank, field_len = self.encode_infobox(batch,
                                                                                                     query_summary)


            # Bridge
            hidden, prior_hidden, mse_loss = self.bridge_init(batch, query_summary, field_summary,
                                                              encoder_memory=(encoder_memory_bank, src_len),
                                                              field_memory=(table_memory_bank[-2], field_len))

            output_probs = torch.zeros(max_len, batch_size, vocab_size, device=device)
            scores = torch.zeros(max_len, batch_size, device=device)
            last_token = torch.ones([batch_size], dtype=torch.long, device=device) * self.tgt_sos_idx
            token_outputs = [last_token]
            ending_flags = torch.zeros([batch_size], dtype=torch.bool, device=device)
            padding_label = torch.ones([batch_size], dtype=torch.long, device=device) * self.tgt_pad_idx
            padding_score = torch.zeros([batch_size], device=device)

            # Generate Equivalent Embeddings And Prepare Embeddings
            equivalent_embeddings, copy_mask, coverage_losses, copy_attention_coverage_vector = \
                self.generate_equivalent_embeddings(batch, encoder_memory_bank, field_representations)

            for t in range(1, max_len):
                prob_output, hidden, attn_weights, copy_attn_weights = self.decoder(
                    last_token, hidden, encoder_memory_bank, table_memory_bank, src_len, field_len,
                    copied_token_equip_embedding=equivalent_embeddings,
                    copy_attention_coverage_vector=copy_attention_coverage_vector,
                    global_node=field_summary,
                )

                if self.src_copy_mode:
                    copy_attn_weights = copy_attn_weights.squeeze(0)
                    coverage_diff = torch.min(copy_attn_weights, copy_attention_coverage_vector)
                    copy_attention_coverage_vector = copy_attention_coverage_vector + copy_attn_weights

                # Ensure the min len
                if min_len != -1 and t <= min_len:
                    prob_output[:, self.tgt_eos_idx] = -1e20

                # Mask unk
                if hparams.disable_unk_output:
                    prob_output[:, self.tgt_unk_idx] = -1e20

                output_probs[t] = prob_output

                # Mask Ended
                score, last_token = prob_output.data.max(1)
                last_token = torch.where(ending_flags, padding_label, last_token)
                token_outputs.append(last_token)
                score = torch.where(ending_flags, padding_score, score)
                scores[t] = score

                # Is finished
                is_finished = last_token == self.tgt_eos_idx
                ending_flags |= is_finished
                if torch.sum(ending_flags) == batch_size:
                    break

            return output_probs, token_outputs, scores


    def get_dialogue_inputs_from_batch(self, batch):
        if not self.table2text_mode:
            src = batch.src[0]
            src_len = batch.src[1]
        else:
            src = None
            src_len = None
        tgt = batch.tgt[0]
        tgt_len = batch.tgt[1]
        batch_size = tgt.size(1)
        return src, tgt, src_len, tgt_len, batch_size

    def encode_query(self, batch):
        src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
        if self.enable_query_encoder is False:
            encoder_memory_bank = None
            hidden = torch.zeros([self.decoder.n_layers, batch_size, self.hidden_size], device=tgt.device)
        else:
            encoder_output, hidden = self.encoder(src, src_len, src_tag=batch.src_tag[0])
            encoder_memory_bank = encoder_output.transpose(0, 1)
            if self.birnn_down_scale is not None:
                encoder_memory_bank = torch.tanh(self.birnn_down_scale(encoder_memory_bank))
        return hidden, encoder_memory_bank

    def bridge_init(self, batch, hidden, infobox_input,
                    encoder_memory=None, field_memory=None):
        mse_loss = None
        prior_hidden = None
        if self.table2text_mode:
            return hidden, prior_hidden, mse_loss

        else:
            # Bridge Part
            if self.bridge_mode == 'general':
                hidden = self.bridge(hidden)
            elif self.bridge_mode == 'attention' or self.bridge_mode == 'field_attention' \
                    or self.bridge_mode == 'fusion_attention':
                hidden, _, _ = self.bridge(hidden, encoder_memory, field_memory)
            elif self.bridge_mode == 'clue_attention' or self.bridge_mode == 'clue_attention2':
                if self.training:
                    src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
                    posterior_output, posterior_hidden = self.encoder(tgt, tgt_len, posterior_mode=True)
                    if self.bridge_mode == 'clue_attention' or self.bridge_mode == 'clue_attention2':
                        query = self.clue_query_projection(self.drop_out(torch.cat([hidden, posterior_hidden], -1)))
                    else:
                        query = self.drop_out(posterior_hidden)
                    posterior_hidden, posterior_attn_weights, posterior_field_attn_weights = \
                        self.posterior_bridge(query, encoder_memory, field_memory)
                    prior_hidden, attn_weights, field_attn_weights = self.bridge(hidden, encoder_memory, field_memory)
                    # => to Batch * Out
                    if self.bridge_mode == 'clue_attention' or self.bridge_mode == 'clue_attention2':
                        mse_loss = 0
                        def kld(p, q):
                            tmp = p * torch.log(p / (q + 1e-20) + 1e-20)
                            tmp = tmp.sum(-1)
                            return tmp
                        for attn_weight, post_attn_weight in zip(attn_weights, posterior_attn_weights):
                            if self.bridge_mode == 'clue_attention':
                                mse_loss += kld(post_attn_weight, attn_weight).mean()
                            elif self.bridge_mode == 'clue_attention2':
                                mse_loss += kld(post_attn_weight, attn_weight).sum()
                        hidden = posterior_hidden
                else:
                    hidden, _, _ = self.bridge(hidden, encoder_memory, field_memory)

            elif self.bridge_mode == 'post_field_attention' \
                    or self.bridge_mode == 'clue_field_attention' or self.bridge_mode == 'clue_field_attention2':
                if self.training:
                    src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
                    posterior_output, posterior_hidden = self.encoder(tgt, tgt_len, posterior_mode=True)
                    if self.bridge_mode == 'clue_field_attention' or self.bridge_mode == 'clue_field_attention2':
                        query = self.clue_query_projection(self.drop_out(torch.cat([hidden, posterior_hidden], -1)))
                    else:
                        query = self.drop_out(posterior_hidden)
                    posterior_hidden, posterior_attn_weights, posterior_field_attn_weights = \
                        self.posterior_bridge(query, encoder_memory,  field_memory)
                    prior_hidden,  attn_weights, field_attn_weights = self.bridge(hidden, encoder_memory, field_memory)
                    # => to Batch * Out
                    if self.bridge_mode == 'post_field_attention':
                        flatten_posterior_hidden = posterior_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
                        flatten_prior_hidden = prior_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
                        mse_loss = flatten_posterior_hidden - flatten_prior_hidden
                        mse_loss = torch.pow(mse_loss, 2)
                        mse_loss = mse_loss.sum()
                    elif self.bridge_mode == 'clue_field_attention' or self.bridge_mode == 'clue_field_attention2':
                        flatten_posterior_hidden = posterior_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
                        flatten_prior_hidden = prior_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
                        mse_loss = flatten_posterior_hidden - flatten_prior_hidden
                        # mse_loss = torch.pow(mse_loss, 2)
                        # mse_loss = mse_loss.sum()
                        mse_loss = 0
                        def kld(p, q):
                            tmp = p * torch.log(p / (q + 1e-20) + 1e-20)
                            tmp = tmp.sum(-1)
                            return tmp
                        for attn_weight, post_attn_weight in zip(field_attn_weights, posterior_field_attn_weights):
                            if self.bridge_mode  == 'clue_field_attention':
                                mse_loss += kld(post_attn_weight, attn_weight).mean()
                            elif self.bridge_mode  == 'clue_field_attention2':
                                mse_loss += kld(post_attn_weight, attn_weight).sum()


                    hidden = posterior_hidden
                else:
                    hidden, _, _ = self.bridge(hidden, encoder_memory, field_memory)
            elif self.bridge_mode[0:6] == 'fusion' or self.bridge_mode == 'latent':
                hidden = self.bridge(hidden, infobox_input)
            elif self.bridge_mode == 'post_fusion' or self.bridge_mode == 'post_fusion2':
                if self.training:
                    src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
                    posterior_output, posterior_hidden = self.encoder(tgt, tgt_len, posterior_mode=True)
                    posterior_hidden = self.posterior_bridge(self.drop_out(posterior_hidden))
                    prior_hidden = self.bridge(hidden, infobox_input)
                    # => to Batch * Out
                    def kld(p, q):
                        tmp = p * torch.log(p / (q + 1e-20) + 1e-20)
                        tmp = tmp.sum()
                        return tmp
                    flatten_posterior_hidden = posterior_hidden.permute(1, 0, 2).view(batch_size, -1)
                    flatten_prior_hidden = prior_hidden.permute(1, 0, 2).view(batch_size, -1)

                    mse_loss = flatten_posterior_hidden - flatten_prior_hidden
                    mse_loss = torch.pow(mse_loss, 2)
                    if self.bridge_mode == 'post_fusion2':
                        mse_loss = mse_loss.sum(-1).mean()
                    else:
                        mse_loss = mse_loss.sum(-1).mean() * 0.5

                    flatten_posterior_hidden = flatten_posterior_hidden / flatten_posterior_hidden.sum(-1, keepdim=True)
                    flatten_prior_hidden = flatten_prior_hidden / flatten_prior_hidden.sum(-1, keepdim=True)
                    mse_loss += 0.5*kld(flatten_posterior_hidden, flatten_prior_hidden)

                    hidden = posterior_hidden
                else:
                    hidden = self.bridge(hidden, infobox_input)


            if self.encoder.rnn_type == 'lstm':
                hidden = [x[:self.decoder.n_layers] for x in hidden]
            else:
                assert len(hidden) == self.decoder.n_layers
                hidden = hidden[:self.decoder.n_layers]

            if prior_hidden is not None:
                if self.encoder.rnn_type == 'lstm':
                    prior_hidden = [x[:self.decoder.n_layers] for x in prior_hidden]
                else:
                    prior_hidden = prior_hidden[:self.decoder.n_layers]

        return hidden, prior_hidden,  mse_loss


    def generate_equivalent_embeddings(self, batch, encoder_memory_bank, field_inputs):
        src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
        max_len = tgt.size(0)

        equip_embeddings = []
        if self.src_copy_mode:
            copy_state_input = [self.decoder.embed(src).transpose(1, 0)]
            copy_mask = model_helper.sequence_mask_fn(src_len, dtype=torch.float32)
            if self.add_pos_tag_embedding:
                copy_state_input.append(self.src_pos_tag_embed(batch.src_tag[0]).transpose(1, 0))
            if self.add_state_to_copy_token:
                copy_state_input.append(encoder_memory_bank)
            copy_state_input = torch.cat(copy_state_input, -1)
            if self.copied_state_to_input_embed_projection is not None:
                copied_token_equip_embedding = self.copied_state_to_input_embed_projection(copy_state_input)
            else:
                copied_token_equip_embedding = copy_state_input
            max_src_len = src.shape[0]
            src_copy_padding_embedding = torch.zeros(
                [batch_size, self.max_copy_token_num - max_src_len, self.embed_size],
                device=copy_state_input.device)
            copied_token_equip_embedding = torch.cat([copied_token_equip_embedding, src_copy_padding_embedding], dim=1)
            equip_embeddings.append(copied_token_equip_embedding)
            coverage_losses = torch.zeros(max_len, device=tgt.device)
            copy_attention_coverage_vector = torch.zeros([batch_size, max_src_len],
                                                         dtype=torch.float32, device=tgt.device)
        else:
            copy_mask = None
            coverage_losses = None
            copy_attention_coverage_vector = None

        if self.field_copy_mode:
            field_state_input = torch.cat(field_inputs, -1).transpose(1, 0)
            field_state_input = self.drop_out(field_state_input)
            field_copy_equip_embedding = self.field_state_to_input_embed_projection(field_state_input)
            equip_embeddings.append(field_copy_equip_embedding)
        if len(equip_embeddings) == 0:
            copied_token_equip_embedding = None
        else:
            copied_token_equip_embedding = torch.cat(equip_embeddings, 1).contiguous()
        return copied_token_equip_embedding, copy_mask, coverage_losses, copy_attention_coverage_vector


    def encode_infobox(self, batch, query_summary):
        if self.enable_field_encoder:
            if self.birnn_down_scale is not None and self.field_encoder.infobox_mode == 'graph':
                query_summary = self.birnn_down_scale(query_summary)
            assert query_summary.size()[0] == 1, 'now only supports 1-layer'
            query_summary = query_summary.squeeze(0)
            field_representations, field_summary, table_memory_bank, field_len = \
                self.field_encoder.encode_infobox(batch, query_summary)
        else:
            field_summary = None
            field_representations = None
            table_memory_bank = (None, None)
            field_len = None
        return field_representations, field_summary, table_memory_bank, field_len


    def forward(self, batch, mode='train'):
        src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
        max_len = tgt.size(0)
        vocab_size_with_offsets = self.decoder.vocab_size_with_offsets
        # Encoding Query
        query_summary, encoder_memory_bank = self.encode_query(batch)
        # Encode Infobox
        field_representations, field_summary, table_memory_bank, field_len = self.encode_infobox(batch, query_summary)
        # Bridge
        hidden, prior_hidden,  mse_loss = self.bridge_init(batch, query_summary, field_summary,
                                                           encoder_memory=(encoder_memory_bank, src_len),
                                                           field_memory=(table_memory_bank[-2], field_len))

        # Generate Equivalent Embeddings And Prepare Embeddings
        equivalent_embeddings, copy_mask, coverage_losses, copy_attention_coverage_vector = \
            self.generate_equivalent_embeddings(batch, encoder_memory_bank, field_representations)

        # Decoder Part
        outputs = torch.zeros(max_len, batch_size, vocab_size_with_offsets, device=tgt.device)

        # Training and Evaluating
        output = tgt.data[0, :]  # 第一个输入 sos
        if mode == 'train':
            token_outputs = []
            teach_force_rate = self.teach_force_rate
        else:
            token_outputs = [output.cpu().numpy()]
            teach_force_rate = 1.0
        if self.decoder.bow_loss and mode == 'train':
            #BOW Representation
            if prior_hidden is not None:
                tmp = prior_hidden.permute(2, 1, 0).reshape(batch_size, -1)
                tmp = self.decoder.dropout(torch.tanh(self.decoder.out_bow_l1(tmp)))
                bow_output = -torch.log(torch.softmax(self.decoder.out_bow_l2(tmp), -1))
                bag_of_words_tgt_index = batch.tgt_bow[0].transpose(1, 0)
                bag_of_words_probs = bow_output.gather(dim=-1, index=bag_of_words_tgt_index)
                valid_index = bag_of_words_tgt_index > 5
                bag_of_words_loss_v2 = torch.where(valid_index, bag_of_words_probs, torch.zeros_like(bag_of_words_probs))
                valid_length = valid_index.sum(dim=-1)
                valid_length = torch.max(valid_length, torch.ones_like(valid_length))
                bag_of_words_loss_v2 = bag_of_words_loss_v2.sum(dim=-1) / valid_length
                bag_of_words_loss = bag_of_words_loss_v2.mean()
            else:
                tmp = hidden.permute(2, 1, 0).reshape(batch_size, -1)
                tmp = self.decoder.dropout(torch.tanh(self.decoder.out_bow_l1(tmp)))
                bow_output = -torch.log(torch.softmax(self.decoder.out_bow_l2(tmp), -1))
                bag_of_words_tgt_index = batch.tgt_bow[0].transpose(1, 0)
                bag_of_words_probs = bow_output.gather(dim=-1, index=bag_of_words_tgt_index)
                valid_index = bag_of_words_tgt_index > 5
                bag_of_words_loss = torch.where(valid_index, bag_of_words_probs, torch.zeros_like(bag_of_words_probs))
                valid_length = valid_index.sum(dim=-1)
                valid_length = torch.max(valid_length, torch.ones_like(valid_length))
                bag_of_words_loss = bag_of_words_loss.sum(dim=-1) / valid_length
                bag_of_words_loss = bag_of_words_loss.mean()

        else:
            bag_of_words_loss = None

        for t in range(1, max_len):
            output, hidden, attn_weights, copy_attn_weights = self.decoder(
                output, hidden, encoder_memory_bank, table_memory_bank, src_len, field_len,
                copied_token_equip_embedding=equivalent_embeddings,
                copy_attention_coverage_vector=copy_attention_coverage_vector,
                global_node=field_summary,
            )
            outputs[t] = output

            _, predict_token = output.data.max(1)
            output = torch.where(torch.rand([batch_size], device=predict_token.device) <= teach_force_rate, tgt.data[t], predict_token)

            if mode == 'eval':
                token_outputs.append(output.cpu().numpy())
            if self.src_copy_mode:
                copy_attn_weights = copy_attn_weights.squeeze(0)
                coverage_diff = torch.min(copy_attn_weights, copy_attention_coverage_vector)
                coverage_loss = (coverage_diff * copy_mask.unsqueeze(1)).sum(-1).mean()
                coverage_losses[t] = coverage_loss
                copy_attention_coverage_vector = copy_attention_coverage_vector + copy_attn_weights
            else:
                copy_attention_coverage_vector = None
        return outputs, token_outputs, coverage_losses, bag_of_words_loss, mse_loss

