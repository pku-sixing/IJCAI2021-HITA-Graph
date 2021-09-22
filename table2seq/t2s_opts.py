import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')

    # Global
    p.add_argument('-mode', type=str, default='train',
                   choices=['train', 'eval', 'infer'],
                   help='running mode')
    p.add_argument('-dataset_version', type=str, default='mid',
                   help='running mode')

    # Train
    p.add_argument('-init_word_vecs', type=str2bool, default=False,
                   help='Use the pre-trained word embedding ')
    p.add_argument('-report_steps', type=int, default=200,
                   help='number of steps for reporting results')
    p.add_argument('-model_path', type=str, default='model')
    p.add_argument('-unk_learning', type=str, default="none",
                   choices=["none", "skip", "penalize"])
    p.add_argument('-cuda', type=str2bool, default=False,
                   help='use cuda or not')
    p.add_argument('-epochs', type=int, default=200,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=64,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='standard learning rate')
    p.add_argument('-init_lr', type=float, default=-1.0,
                   help='initial learning rate, -1.0 is disable')
    p.add_argument('-init_lr_decay_epoch', type=int, default=6,
                   help='number of epochs for train')

    p.add_argument('-lr_decay_rate', type=float, default=0.5,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=5.0,
                   help='in case of gradient explosion')

    # Infobox
    p.add_argument('-field_key_vocab_path', type=str, default='data/field_vocab.txt',
                   help='field_key_vocab_path')
    p.add_argument('-src_tag_vocab_path', type=str, default='data/pos_tag.txt',
                   help='pos_tag key vocab_path')
    p.add_argument('-add_pos_tag_embedding', type=str2bool, default=False,
                   help='add_pos_tag_embedding')
    p.add_argument('-enable_field_attn', type=str2bool, default=True,
                   help='enable_field_attn')
    p.add_argument('-enable_query_attn', type=str2bool, default=True,
                   help='enable_field_attn')

    p.add_argument('-field_word_vocab_path', type=str, default='none',
                   help='field word vocab_path')
    p.add_argument('-field_word_tag_path', type=str, default='none',
                   help='field word pos tag vocab_path')
    p.add_argument('-field_key_tag_path', type=str, default='none',
                   help='field word pos tag vocab_path')
    p.add_argument('-field_key_embed_size', type=int, default=200,
                   help='field key embedding size')
    p.add_argument('-max_kw_pairs_num', type=int, default=300,
                   help='max_kw_pairs_num')
    p.add_argument('-max_kv_pairs_num', type=int, default=50,
                   help='max_kv_pairs_num')
    p.add_argument('-max_field_intra_word_num', type=int, default=200,
                   help='max_field_intra_word_num')
    p.add_argument('-field_position_embedding_size', type=int, default=5,
                   help='field key embedding size')
    p.add_argument('-field_tag_embedding_size', type=int, default=10,
                   help='field key embedding size')
    # transformer_field_encoder
    p.add_argument('-transformer_field_encoder_heads', type=int, default=4,
                   help='transformer_field_encoder_heads')
    p.add_argument('-transformer_field_encoder_layers', type=int, default=2,
                   help='transformer_field_encoder_layers')
    # hierarchical_field_encoder and hierarchical_intra_field_encoder
    p.add_argument('-hierarchical_field_encoder_heads', type=int, default=4,
                   help='hierarchical_field_encoder_heads')
    # hierarchical_infobox_encoder
    p.add_argument('-hierarchical_infobox_encoder_heads', type=int, default=4,
                   help='hierarchical_infobox_encoder_heads')
    p.add_argument('-hierarchical_infobox_rgat_layers', type=int, default=1,
                   help='hierarchical_infobox_rgat_layers')
    p.add_argument('-hierarchical_infobox_rgat_learnable_global_node', type=bool, default=True,
                   help='hierarchical_infobox_rgat_learnable_global_node')
    p.add_argument('-hierarchical_infobox_rgat_relational_weights', type=bool, default=False,
                   help='hierarchical_infobox_rgat_relational_weights')
    p.add_argument('-hierarchical_infobox_attention', type=bool, default=False,
                   help='hierarchical_infobox_attention')
    p.add_argument('-hierarchical_infobox_attention_type',
              type=str, default='mlp',
              choices=['mlp', 'general'], help="")

    p.add_argument('-field_copy', type=str2bool, default=False,
                   help='copy field values')
    p.add_argument('-copy_query_first', type=str2bool, default=False,
                   help='first copy query')
    p.add_argument('-dual_field_word_embedding', type=str2bool, default=False,
                   help='use word embedding and the field word embedding at the same time')

    p.add_argument('-dual_attn',
              type=str, default='none',
              choices=['none', 'general', 'gate', "selector", "gate_fusion"],
              help="")
    p.add_argument('-infobox_memory_bank_format',
              type=str, default='fw_fk',
              choices=['none', 'fw_fk', 'fwk_fwv_fk'],
              help="")
    p.add_argument('-field_tag_usage',
              type=str, default='none',
              choices=['none', 'general'],
              help="")
    p.add_argument('-field_input_tags', type=str, default='local_pos_fw,local_pos_bw', help="")

    # Dataset
    p.add_argument('-train_data_path_prefix', type=str, default='data/test',
                   help='the data path and prefix for the training files')
    p.add_argument('-val_data_path_prefix', type=str, default='data/test',
                   help='the data path and prefix for the validation files')
    p.add_argument('-test_data_path_prefix', type=str, default='data/test',
                   help='the data path and prefix for the test files')
    p.add_argument('-vocab_path', type=str, default='data/vocab.txt',
                   help='vocab_path, available if share_vocab')
    p.add_argument('-src_vocab_path', type=str, default='data/vocab.txt',
                   help='src_vocab_path, available if not share_vocab')
    p.add_argument('-tgt_vocab_path', type=str, default='data/vocab.txt',
                   help='tgt_vocab_path, available if not share_vocab')
    p.add_argument('-share_vocab', type=str2bool, default=True,
                   help='src and tgt share a vocab')
    p.add_argument('-share_embedding', type=str2bool, default=True,
                   help='src and tgt share an embedding')
    p.add_argument('-random_train_field_order', type=str2bool, default=False,
                   help='src and tgt share an embedding')
    p.add_argument('-random_test_field_order', type=str2bool, default=False,
                   help='src and tgt share an embedding')
    p.add_argument('-load_augmented_data', type=str, default="none",
                   choices=["none", "general", "src_only","tgt_only","all"])
    p.add_argument('-max_src_len', type=int, default=35,
                   help='max_src_len')
    p.add_argument('-max_tgt_len', type=int, default=35,
                   help='max_tgt_len')
    p.add_argument('-max_line', type=int, default=-1,
                   help='max loaded line number, -1 means do not restrict')


    # Infer
    p.add_argument('-repeat_index', type=int, default=0,
                   help='repeat_number')
    p.add_argument('-infer_max_len', type=int, default=35,
                   help='infer_max_len')
    p.add_argument('-infer_min_len', type=int, default=3,
                   help='infer_min_len')
    p.add_argument('-penalize_repeat_tokens_rate', type=float, default=2.0,
                   help='')
    p.add_argument('-disable_multiple_copy', type=str2bool, default=True,
                   help='')
    p.add_argument('-disable_unk_output', type=str2bool, default=False,
                   help='mask the output of unks')
    p.add_argument('-use_best_model', type=str2bool, default=True,
                   help='mask the output of unks')
    p.add_argument('-skip_infer', type=str2bool, default=False,
                   help='only evaluating the result')
    p.add_argument('-beam_width', type=int, default=10,
                   help='beam search width')
    p.add_argument('-diverse_decoding', type=float, default=0.0,
                   help='mask the output of unks')
    p.add_argument("--pre_embed_file", type=str, default="/home/sixing/dataset/embed/tencent.txt",
                        help="enable binary selector")
    p.add_argument("--pre_embed_dim", type=int, default=200, help="enable binary selector")
    p.add_argument("--subword", type=str, default=None)
    p.add_argument('-beam_length_penalize', type=str, default='avg',
                   choices=['none', 'avg'], help='beam_length_penalize')
    # Model
    p.add_argument('-bow_loss', type=float, default=0.0,
                   help='bow_loss rate')
    p.add_argument('-dropout', type=float, default=0.5,
                   help='dropout value')
    p.add_argument('-align_dropout', type=float, default=0.0,
                   help='dropout value')
    p.add_argument('-hidden_size', type=int, default=512,
                   help='size of a hidden layer')
    p.add_argument('-embed_size', type=int, default=300, help='size of word embeddings')
    p.add_argument('-rnn_type', type=str, default='gru',
                   choices=['lstm', 'gru'],
                   help='size of word embeddings')
    p.add_argument('-init', type=str, default='xavier',
                   choices=['xavier', 'xavier_normal', 'uniform', 'kaiming', 'kaiming_normal'],
                   help='size of word embeddings')

    # Encoder
    p.add_argument('-enc_layers', type=int, default=1,
                   help='number of encoder layers')
    p.add_argument('-enc_birnn', type=str2bool, default=True,
                   help='use cuda or not')

    # Table Field
    p.add_argument('-field_encoder', type=str, default='lstm',
                   choices=['lstm', 'transformer', 'hierarchical_lstm', 'hierarchical_field', 'hierarchical_intra_field', 'hierarchical_infobox'],
                   help='type of field encoder')
    p.add_argument('-task_mode', type=str, default='table_dialogue',
                   choices=['table_dialogue', 'text2text','table2text'],
                   help='task_mode')

    # Bridge
    p.add_argument('-bridge', type=str, default='general',
                   choices=['none', 'general','fusion','fusion2',
                            'attention','field_attention','clue_field_attention',
                            'clue_attention','clue_attention2',
                            'fusion_attention','post_attention_fusion','clue_field_attention2'],
                   help='size of word embeddings')
    # CharEncoders
    p.add_argument('-char_encoders', type=str, default='none',
                   help='CharEncoders')
    p.add_argument('-char_encoder_type', type=str, default='none',
                   help='CharEncoders')
    p.add_argument('-char_encoder_share_mode', type=str, default='separated',
                   help='CharEncoders')
    p.add_argument('-char_vocab_path', type=str, default='separated',
                   help='CharEncoders')

    # Decoder
    p.add_argument('-mode_selector',
              type=str, default='general',
              choices=['general', 'mlp'],
              help="Mode Selector")
    p.add_argument('-attn_type',
              type=str, default='general',
              choices=['dot', 'general', 'mlp', 'none'],
              help="The attention type to use: "
                   "dotprod or general (Luong) or MLP (Bahdanau)")
    p.add_argument( '-attn_func',
              type=str, default="softmax", choices=["softmax", "sparsemax"])
    p.add_argument('-dec_layers', type=int, default=1,
                   help='number of decoder layers')

    p.add_argument('-teach_force_rate', type=float, default=1.0,
                   help='teach_force_rate, 1.0 means use the ground truth, 0.0 means use the predicted tokens')
    p.add_argument('-copy', type=str2bool, default=False,
                   help='enable pointer-copy')
    p.add_argument('-complex_attention_query', type=str2bool, default=False,
                   help='complex_attention_query')
    p.add_argument('-update_decoder_with_global_node', type=str2bool, default=False,
                   help='complex_attention_query')
    p.add_argument('-add_state_to_copy_token', type=str2bool, default=False,
                   help='enable pointer-copy')
    p.add_argument('-share_copy_attn', type=str2bool, default=False,
                   help='sharing src/field copy attention function')
    p.add_argument('-share_field_copy_attn', type=str2bool, default=False,
                   help='sharing src/field copy attention function')
    p.add_argument('-add_last_generated_token', type=str2bool, default=False,
                   help='add_last_generated_token')
    p.add_argument('-max_copy_token_num', type=int, default=40,
                   help='it should be greater than maximum src len')
    p.add_argument('-copy_coverage', type=float, default=-1,
                   help='copy coverage loss rate, 0.0 = disabled')
    p.add_argument('-copy_attn_type',
              type=str, default='mlp',
              choices=['dot', 'general', 'mlp', 'none'],
              help="The copy attention type to use: "
                   "dotprod or general (Luong) or MLP (Bahdanau)")
    p.add_argument( '-copy_attn_func',
              type=str, default="softmax", choices=["softmax", "sparsemax"])


    return p.parse_args()

