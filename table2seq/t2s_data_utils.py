import torchtext
from collections import Counter, OrderedDict
from torchtext.data import Field, BucketIterator, Dataset, Iterator

from utils.text.GraphField import GraphField
from utils.text.SubTextField import SubTextField
from torchtext.vocab import Vocab
from utils.logger import logger
from utils.text.tokenization import get_tokenizer
import re
import torch
import random
import numpy as np
import pickle

SRC_SUFFIX = 'src'
BOX_SUFFIX = 'box'
TGT_SUFFIX = 'tgt'
TAG_SUFFIX = 't'
STOP_COPY_BEFORE_POSITION = 300


def load_pretrain_embeddings(embedding, filed, embedding_path, dim=200, char2word=None, suffix=None):
    total_word = len(filed.vocab)
    flag = True
    loaded_vecs = None
    if suffix is not None:
        logger.info('[PRE-EMBED] Try to load embedding for %s from the cache %s.embed' % (str(filed), suffix))
        try:
            loaded_vecs = pickle.load(open('embed_cache/%s.embed' % suffix, 'rb'))
            logger.info('[PRE-EMBED] Successfully loaded embedding for %s from the cache .%s' % (str(filed), suffix))
        except FileNotFoundError:
            loaded_vecs = None
            logger.info('[PRE-EMBED] Failed to load embedding for %s from the cache .%s' % (str(filed), suffix))

    if loaded_vecs is None:
        loaded_vecs = dict()
        logger.info('[PRE-EMBED] Loading for %s, Char2Word: %s' % (str(filed), str(char2word)))
        token_set = set()
        for word in filed.vocab.stoi:
            token_set.add(word)
            if char2word is not None:
                for char in word:
                    token_set.add(char)

        with open(embedding_path, 'r+', encoding='utf-8') as fin:
            line = fin.readline()
            while len(line) > 0:
                items = line.strip('\r\n').split()
                if len(items) != dim + 1:
                    line = fin.readline()
                    continue
                word = items[0]
                weights = items[1:]
                if word in token_set:
                    weights = np.array([float(x) for x in weights])
                    loaded_vecs[word] = weights
                    flag = True
                if len(loaded_vecs) % 1000 == 0 and flag:
                    logger.info('[PRE-EMBED] Loading: %d/%d' % (len(loaded_vecs), total_word))
                    flag = False
                line = fin.readline()
        if suffix is not None:
            pickle.dump(loaded_vecs, open('embed_cache/%s.embed' % suffix, 'wb'),)
    logger.info('[PRE-EMBED] Loaded Token/Total: %d/%d' % (len(loaded_vecs), total_word))
    pretrained_weight = np.zeros([total_word, dim])
    weights = embedding.weight.data.cpu().numpy()

    load_count = 0
    generate_count = 0
    for i in range(total_word):
        word = filed.vocab.itos[i]
        if word in loaded_vecs:
            load_count += 1
            pretrained_weight[i] = loaded_vecs[word]
        else:
            if char2word is None:
                pretrained_weight[i] = weights[i]
            elif char2word == 'avg':
                tmp = np.zeros([dim])
                tmp_flag = False
                for char in word:
                    if char in loaded_vecs:
                        tmp += loaded_vecs[char]
                        tmp_flag = True
                    else:
                        tmp += weights[i]

                if tmp_flag:
                    generate_count += 1
                    tmp /= len(word)
                    pretrained_weight[i] = tmp
                else:
                    pretrained_weight[i] = weights[i]
            else:
                raise NotImplementedError()

    logger.info('[PRE-EMBED] Loaded/Generated/Word/Total: %d/%d/%d' % (load_count, generate_count, total_word))
    is_cuda = embedding.weight.device
    embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
    embedding.weight.to(is_cuda)


def translate(ids, field):
    if ids is None or field is None:
        return None
    trans_res = []
    max_time = len(ids)
    batch_num = len(ids[0])
    for batch in range(batch_num):
        res = ' '.join([field.vocab.itos[ids[t][batch]] for t in range(max_time)])
        res = re.sub('<[(sos|pad|eos)]+>', '', res)
        res = re.sub('<[ ]+>', ' ', res)
        trans_res.append(res.strip())
    return trans_res


def _process_copy_examples(srcs, boxes, tgts, has_sos=True, word_freq_dict=None, query_first=False, dropout=0.0):
    num_total = len(tgts)
    word_types = []
    for idx in range(num_total):
        tgt = tgts[idx].split()
        if srcs is not None:
            src = srcs[idx].split()
        else:
            src = None
        if boxes is not None:
            box = boxes[idx]
        else:
            box = None

        src_map = dict()
        offset = 1 if has_sos else 0
        table_offset = 0

        if query_first is False:
            if src is not None:
                # 首先增加Copy的占位符，随后再增加Table的
                rand_flags = np.random.random([len(src)])
                if word_freq_dict is None:
                    for i, word in enumerate(src):
                        if rand_flags[i] < dropout:
                            continue
                        src_map[word] = '<src_%d>' % (i + offset)
                else:
                    for i, word in enumerate(src):
                        if rand_flags[i] < dropout:
                            continue
                        if word_freq_dict.get(word, STOP_COPY_BEFORE_POSITION + 1) > STOP_COPY_BEFORE_POSITION:
                            src_map[word] = '<src_%d>' % (i + offset)

            if box is not None:
                # 增加Table的
                box = box.strip('\r\n')
                field_values = []
                for field in box.split(' '):
                    items = field.split(':')
                    field_value = items[1]
                    field_values.append(field_value)
                rand_flags = np.random.random([len(field_values)])
                if word_freq_dict is None:
                    for i, word in enumerate(field_values):
                        if rand_flags[i] < dropout:
                            continue
                        src_map[word] = '<field_%d>' % (i + table_offset)
                else:
                    for i, word in enumerate(field_values):
                        if rand_flags[i] < dropout:
                            continue
                        if word_freq_dict.get(word, -1) > STOP_COPY_BEFORE_POSITION:
                            src_map[word] = '<field_%d>' % (i + table_offset)
        else:
            if box is not None:
                # 增加Table的
                box = box.strip('\r\n')
                field_values = []
                for field in box.split(' '):
                    items = field.split(':')
                    field_value = items[1]
                    field_values.append(field_value)
                rand_flags = np.random.random([len(field_values)])
                if word_freq_dict is None:
                    for i, word in enumerate(field_values):
                        if rand_flags[i] < dropout:
                            continue
                        src_map[word] = '<field_%d>' % (i + table_offset)
                else:
                    for i, word in enumerate(field_values):
                        if rand_flags[i] < dropout:
                            continue
                        if word_freq_dict.get(word, -1) > STOP_COPY_BEFORE_POSITION:
                            src_map[word] = '<field_%d>' % (i + table_offset)

            if src is not None:
                # 首先增加Copy的占位符，随后再增加Table的
                rand_flags = np.random.random([len(src)])
                if word_freq_dict is None:
                    for i, word in enumerate(src):
                        if rand_flags[i] < dropout:
                            continue
                        src_map[word] = '<src_%d>' % (i + offset)
                else:
                    for i, word in enumerate(src):
                        if rand_flags[i] < dropout:
                            continue
                        if word_freq_dict.get(word, STOP_COPY_BEFORE_POSITION + 1) > STOP_COPY_BEFORE_POSITION:
                            src_map[word] = '<src_%d>' % (i + offset)



        word_type = []
        for i, word in enumerate(tgt):
            tgt[i] = src_map.get(word, word)
        tgts[idx] = ' '.join(tgt)
        word_types.append(word_type)
    return word_types


def restore_field_copy_example(field_values, src, tgt, has_sos=True):
    if field_values is None:
        return restore_pointer_copy_example(src, tgt, has_sos)
    field_values = field_values.split()
    src = src.split()
    tgt = tgt.split()
    src_map = dict()
    offset = 1 if has_sos else 0
    for i, word in enumerate(src):
        src_map['<src_%d>' % (i + offset)] = word
    table_offset = 0
    for i, word in enumerate(field_values):
        src_map['<field_%d>' % (i + table_offset)] = word

    for i, word in enumerate(tgt):
        tgt[i] = src_map.get(word, word)
    return ' '.join(tgt)


def restore_pointer_copy_example(src, tgt, has_sos=True):
    src = src.split()
    tgt = tgt.split()
    src_map = dict()
    offset = 1 if has_sos else 0
    for i, word in enumerate(src):
        src_map['<src_%d>' % (i + offset)] = word
    for i, word in enumerate(tgt):
        tgt[i] = src_map.get(word, word)
    return ' '.join(tgt)


def load_examples(example_path, field_words=False):
    with open(example_path + '.' + SRC_SUFFIX, 'r', encoding='utf-8') as fin:
        src = fin.readlines()
    logger.info('[DATASET] Loaded %s.src, lines=%d' % (example_path, len(src)))
    with open(example_path + '.' + TGT_SUFFIX, 'r', encoding='utf-8') as fin:
        tgt = fin.readlines()
    logger.info('[DATASET] Loaded %s.tgt, lines=%d' % (example_path, len(tgt)))

    src = [x.strip('\r\n') for x in src]
    tgt = [x.strip('\r\n') for x in tgt]

    if field_words is False:
        return src, tgt
    else:
        with open(example_path + '.' + BOX_SUFFIX, 'r', encoding='utf-8') as fin:
            boxes = fin.readlines()
        logger.info('[DATASET] Loaded %s.box, lines=%d' % (example_path, len(boxes)))
        fields = []
        for box in boxes:
            box = box.strip('\r\n')
            field_values = []
            for field in box.split(' '):
                items = field.split(':')
                field_value = items[1]
                field_values.append(field_value)
            fields.append(' '.join(field_values))
        return src, tgt, fields


def get_dataset(hparams, example_path, field_orders, field_dict, copy_dropout=0.0, random_order=False):
    load_augmented_data = hparams.load_augmented_data
    max_line = hparams.max_line
    max_src_len = hparams.max_src_len
    max_tgt_len = hparams.max_tgt_len
    max_kw_pairs_num = hparams.max_kw_pairs_num
    pointer_copy = hparams.copy
    field_copy = hparams.field_copy
    assert load_augmented_data == 'none'
    if pointer_copy:
        assert 'src' in field_dict
    if field_copy:
        assert 'attribute_key' in field_dict

    # store
    my_examples = dict()

    # SRC
    if 'src' in field_dict:
        with open(example_path + '.' + SRC_SUFFIX, 'r', encoding='utf-8') as fin:
            src = fin.readlines()
        logger.info('[DATASET] Loaded %s.src, lines=%d' % (example_path, len(src)))
        my_examples['src'] = src


        # SRC TAG
        if 'src_tag' in field_dict:
            with open(example_path + '.' + SRC_SUFFIX + TAG_SUFFIX, 'r', encoding='utf-8') as fin:
                src_tag = fin.readlines()
            logger.info('[DATASET] Loaded tags %s.srct, lines=%d' % (example_path, len(src_tag)))
            my_examples['src_tag'] = src_tag
            src_tag = None

    # Infobox
    if 'attribute_key' in field_dict:
        with open(example_path + '.' + BOX_SUFFIX, 'r', encoding='utf-8') as fin:
            tmp = fin.readlines()
            box = []
            for item in tmp:
                items = ' '.join(item.strip('\r\n').split(' ')[0:max_kw_pairs_num - 1])
                box.append(items)
        logger.info('[DATASET] Loaded %s.box, lines=%d' % (example_path, len(box)))
        my_examples['box'] = box
        box = None

    # TGT
    with open(example_path + '.' + TGT_SUFFIX, 'r', encoding='utf-8') as fin:
        tgt = fin.readlines()
        logger.info('[DATASET] Loaded %s.tgt, lines=%d' % (example_path, len(tgt)))
        my_examples['tgt'] = tgt
        tgt = None

    # TGT_RAW
    with open(example_path + '.' + TGT_SUFFIX, 'r', encoding='utf-8') as fin:
        tgt_bow = fin.readlines()
        logger.info('[DATASET-BOW] Loaded %s.tgt, lines=%d' % (example_path, len(tgt_bow)))
        my_examples['tgt_bow'] = tgt_bow
        tgt_bow = None

    # 裁剪一部分数据
    if max_line != -1:
        for idx in my_examples.keys():
            my_examples[idx] = my_examples[idx][0:max_line]

    if max_src_len != -1 and 'src' in field_dict:
        logger.info('[DATASET] Maximum source sequence length is set to %d' % (max_src_len))
        my_examples['src'] = [' '.join(x.strip('\r\n').split()[0:max_src_len]) for x in my_examples['src']]
        src_max_len = max([len(x.split()) for x in my_examples['src']])
    else:
        src_max_len = -1

    if max_tgt_len != -1:
        logger.info('[DATASET] Maximum target sequence length is set to %d' % (max_tgt_len))
        my_examples['tgt'] = [' '.join(x.strip('\r\n').split()[0:max_tgt_len]) for x in my_examples['tgt']]

    copy_mode = []
    if pointer_copy:
        copy_mode.append('src')
    if field_copy:
        copy_mode.append('field')
    copy_mode = '-'.join(copy_mode)

    if copy_mode == 'src':
        logger.info('[DATASET] Aligning src -> tgt for Pointer-Copy ')
        word_freq_dict = field_dict['tgt'].vocab.stoi
        word_types = _process_copy_examples(my_examples['src'], None, my_examples['tgt'],
                                            query_first=hparams.copy_query_first,
                                            word_freq_dict=word_freq_dict, dropout=copy_dropout)
    elif copy_mode == 'src-field':
        logger.info('[DATASET] Aligning src & infobox -> tgt for Pointer-Copy ')
        word_freq_dict = field_dict['tgt'].vocab.stoi
        word_types = _process_copy_examples(my_examples['src'], my_examples['box'], my_examples['tgt'],
                                            query_first=hparams.copy_query_first,
                                            word_freq_dict=word_freq_dict, dropout=copy_dropout)
    elif copy_mode == 'field':
        logger.info('[DATASET] Aligning infobox -> tgt for Pointer-Copy ')
        word_freq_dict = field_dict['tgt'].vocab.stoi
        word_types = _process_copy_examples(None, my_examples['box'], my_examples['tgt'],
                                            query_first=hparams.copy_query_first,
                                            word_freq_dict=word_freq_dict, dropout=copy_dropout)

    def _box_split(z, sub_key=False, sub_word=False, random_order=False):
        z = z.strip('\r\n')
        field_keys = []
        sub_field_keys =[]
        pos_sts = []
        pos_eds = []
        pos_kv = []
        pos_kw = []
        tags = []
        field_values = []
        sub_field_values = []

        key_position = 0
        previous_start = -1
        previous_key = '<none>'
        kv_list = [key_position]
        field_items = z.split(' ')
        if random_order:
            if len(field_items) == 1:
                pass
            else:
                field_groups = []
                tmp_group = []
                for field in field_items[1:]:
                    items = field.split(':')
                    field_key, pos_st, pos_ed, tag = items[0].split('_')
                    pos_st = int(pos_st)
                    pos_ed = int(pos_ed)
                    tmp_group.append(field)
                    if pos_ed == 1:
                        field_groups.append(tmp_group)
                        tmp_group = []
                field_groups = random.sample(field_groups, len(field_groups))
                new_items = [field_items[0]]
                for group in field_groups:
                    new_items += group
                field_items = new_items

        for field in field_items:
            items = field.split(':')
            field_key, pos_st, pos_ed, tag = items[0].split('_')
            pos_st = int(pos_st)
            pos_ed = int(pos_ed)

            if pos_st != previous_start + 1 or previous_key != field_key:
                key_position += 1
                kv_list.append(key_position)
            previous_start = pos_st
            previous_key = field_key
            pos_kv.append(key_position)
            pos_kw.append(len(pos_kw))

            field_value = items[1]
            field_keys.append(field_key)
            sub_field_keys.append(field_key+'')
            pos_sts.append(pos_st)
            pos_eds.append(pos_ed)
            tags.append(tag)
            field_values.append(field_value)
            sub_field_values.append(field_value+'')

        graph_node_number = len(kv_list)

        # 3 keys
        res = [graph_node_number, field_keys]
        if sub_key:
            res += [sub_field_keys]
        res += [field_keys, field_keys, field_values]
        if sub_word:
            res += [sub_field_values]
        res +=[field_values, tags, pos_kv, pos_kw, pos_sts, pos_eds]
        return res

    # 创造初始的输入
    lines = []
    for a_idx in my_examples.keys():
        for b_idx in my_examples.keys():
            assert len(my_examples[a_idx]) == len(my_examples[b_idx])
    raw_num = len(my_examples['tgt'])
    for idx in range(raw_num):
        line = []
        if 'src' in my_examples:
            line += [my_examples['src'][idx].strip('\r\n')]
        if 'sub_src' in field_dict:
            line += [my_examples['src'][idx].strip('\r\n')]
        if 'src_tag' in my_examples:
            line += [my_examples['src_tag'][idx].strip('\r\n')]
        line += [my_examples['tgt'][idx].strip('\r\n'), my_examples['tgt_bow'][idx].strip('\r\n')]
        if 'box' in my_examples:
            line += _box_split(my_examples['box'][idx], 'sub_attribute_key' in field_dict,
                               'sub_attribute_word' in field_dict, random_order=random_order)
        lines.append(line)
        assert len(line) == len(field_orders), '%s-%s' % (len(line), len(field_orders))

    examples = []
    for line in lines:
        example = torchtext.data.Example.fromlist(data=line, fields=field_orders)
        examples.append(example)
    dataset = Dataset(examples=examples, fields=field_orders)
    return dataset, src_max_len


def load_vocab(vocab_path, field, special_tokens=[], pointer_copy_tokens=0, field_copy_tokens=0):
    counter = Counter()
    with open(vocab_path, 'r+', encoding='utf-8') as fin:
        vocabs = [x.strip('\r\n') for x in fin.readlines()]
        if pointer_copy_tokens > 0:
            vocabs += ['<src_%d>' % x for x in range(pointer_copy_tokens)]
        if field_copy_tokens > 0:
            vocabs += ['<field_%d>' % x for x in range(field_copy_tokens)]
        vocab_size = len(vocabs)
        for idx, token in enumerate(vocabs):
            counter[token] = vocab_size - idx
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token] + special_tokens
        if tok is not None))
    field.vocab = Vocab(counter, specials=specials)


def load_dataset(hparams, is_eval=False, test_data_path=None):
    batch_size = hparams.batch_size
    max_copy_token_num = hparams.max_copy_token_num
    pointer_copy_tokens = hparams.max_copy_token_num if hparams.copy else 0
    field_copy_tokens = hparams.max_kw_pairs_num if hparams.field_copy else 0

    enabled_char_encoders = set(hparams.char_encoders.split(','))
    assert len(enabled_char_encoders - set('field_key,field_word,src,none'.split(','))) == 0, hparams.char_encoders

    tokenize = get_tokenizer('default')

    # Dialogue
    src_field = Field(tokenize=tokenize, include_lengths=True,
                      init_token='<ssos>', eos_token='<seos>')
    sub_src_field = SubTextField(tokenize=tokenize, include_lengths=True,
                                 init_token='<ssos>', eos_token='<seos>')
    src_tag_field = Field(tokenize=tokenize, include_lengths=True,
                          init_token='<ssos>', eos_token='<seos>')
    tgt_field = Field(tokenize=tokenize, include_lengths=True,
                      init_token='<sos>', eos_token='<eos>')
    tgt_bow_field = Field(tokenize=tokenize, include_lengths=True,
                          init_token='<sos>', eos_token='<eos>')

    # Infobox
    attribute_graph_field = GraphField(tokenize=None, include_lengths=True,
                                init_token=None, eos_token='<seos>')
    attribute_key_field = Field(tokenize=None, include_lengths=True,
                                init_token=None, eos_token='<seos>')
    sub_attribute_key_field = SubTextField(tokenize=None, include_lengths=True,
                                           init_token=None, eos_token='<seos>')
    attribute_uni_key_field = Field(tokenize=None, include_lengths=True,
                                    init_token=None, eos_token='<seos>')
    attribute_key_tag_field = Field(tokenize=None, include_lengths=True,
                                    init_token=None, eos_token='<seos>', )
    attribute_word_field = Field(tokenize=None, include_lengths=True,
                                 init_token=None, eos_token='<seos>', )
    sub_attribute_word_field = SubTextField(tokenize=None, include_lengths=True,
                                     init_token=None, eos_token='<seos>', )
    attribute_uni_word_field = Field(tokenize=None, include_lengths=True,
                                     init_token=None, eos_token='<seos>', )
    attribute_word_tag_field = Field(tokenize=None, include_lengths=True,
                                     init_token=None, eos_token='<seos>', )
    attribute_kv_pos_field = Field(tokenize=None, include_lengths=True, use_vocab=False,
                                   init_token=None, eos_token=0, pad_token=0, unk_token=0)
    attribute_kw_pos_field = Field(tokenize=None, include_lengths=True, use_vocab=False,
                                   init_token=None, eos_token=0, pad_token=0, unk_token=0)

    attribute_word_local_fw_pos_field = Field(tokenize=None, include_lengths=True, use_vocab=False,
                                              init_token=None, eos_token=0, pad_token=0, unk_token=0)
    attribute_word_local_bw_pos_field = Field(tokenize=None, include_lengths=True, use_vocab=False,
                                              init_token=None, eos_token=0, pad_token=0, unk_token=0)

    fields = []
    sub_fields = []
    logger.info('[TASK-MODE]  %s' % hparams.task_mode)
    if hparams.task_mode != 'table2text':
        fields += [('src', src_field)]
        if 'src' in enabled_char_encoders:
            logger.info('[CHAR_ENCODER] add src char_encoder')
            fields += [('sub_src', sub_src_field)]
            sub_fields += [('sub_src', sub_src_field)]
        fields += [ ('src_tag', src_tag_field)]
    fields += [('tgt', tgt_field), ('tgt_bow', tgt_bow_field)]
    if hparams.task_mode != 'text2text':
        fields += [('attribute_graph', attribute_graph_field)]
        fields += [('attribute_key', attribute_key_field)]
        if 'field_key' in enabled_char_encoders:
            logger.info('[CHAR_ENCODER] src_attribute_key  char_encoder')
            fields += [('sub_attribute_key', sub_attribute_key_field)]
            sub_fields += [('sub_attribute_key', sub_attribute_key_field)]
        fields += [('attribute_uni_key', attribute_uni_key_field),
                   ('attribute_key_tag', attribute_key_tag_field)]
        fields += [('attribute_word', attribute_word_field)]
        if 'field_word' in enabled_char_encoders:
            logger.info('[CHAR_ENCODER] add src_attribute_word char_encoder')
            fields += [('sub_attribute_word', sub_attribute_word_field)]
            sub_fields += [('sub_attribute_word', sub_attribute_word_field)]
        fields += [('attribute_uni_word', attribute_uni_word_field),
                   ('attribute_word_tag', attribute_word_tag_field)]
        fields += [('attribute_kv_pos', attribute_kv_pos_field), ('attribute_kw_pos', attribute_kw_pos_field)]
        fields += [('attribute_word_local_fw_pos', attribute_word_local_fw_pos_field),
                   ('attribute_word_local_bw_pos', attribute_word_local_bw_pos_field)]

    # To Field Dict
    field_dict = {}
    for x in fields:
        field_dict[x[0]] = x[1]

    # SubVocabs
    if hparams.char_encoders != 'none':
        logger.info('[VOCAB] Constructing char vocabs for char_encoders')
        sp_tokens = ['<w>', '</w>','<p>']
        if hparams.char_encoder_share_mode == 'none':
            logger.info('[VOCAB] Constructing one char vocab for all encoders')
            last_field = None
            for field in sub_fields:
                if last_field != None:
                    logger.info('[VOCAB] reuse %s\'s char vocab for %s' % (str(last_field[0]), str(field[0])))
                    field[1].vocab = last_field[1].vocab
                    logger.info('[VOCAB] %s char vocab size: %d' % (str(field[0]), len(field[1].vocab.itos)))
                else:
                    logger.info('[VOCAB] Loading src char vocab from: %s' % hparams.char_vocab_path)
                    load_vocab(hparams.char_vocab_path, field[1], special_tokens=sp_tokens)
                    logger.info('[VOCAB] %s char vocab size: %d' % (str(field[0]), len(field[1].vocab.itos)))
                    last_field = field
        elif hparams.char_encoder_share_mode == 'separated':
            char_vocab_map = {
                'sub_src' : hparams.vocab_path + '.char',
                'sub_attribute_key' : hparams.field_key_vocab_path + '.char',
                'sub_attribute_word' : hparams.field_word_vocab_path + '.char',
            }
            logger.info('[VOCAB] Constructing one char vocab for one char encoder')
            for field in sub_fields:
                char_vocab_path = char_vocab_map[field[0]]
                logger.info('[VOCAB] Loading src char vocab from: %s' % char_vocab_path)
                load_vocab(char_vocab_path , field[1], special_tokens=sp_tokens)
                logger.info('[VOCAB] %s char vocab size: %d' % (str(field[0]), len(field[1].vocab.itos)))
        else:
            raise NotImplementedError()
    # Dialogue Vocabs
    if not hparams.share_vocab:
        logger.info('[VOCAB] Constructing two vocabs for the src and tgt')
        if 'src' in field_dict:
            logger.info('[VOCAB] Loading src vocab from: %s' % hparams.src_vocab_path)
            load_vocab(hparams.src_vocab_path, src_field)
            logger.info('[VOCAB] src vocab size: %d' % len(src_field.vocab.itos))
        else:
            logger.info('[VOCAB] Ignored src vocab')
        logger.info('[VOCAB] Loading tgt vocab from: %s' % hparams.tgt_vocab_path)
        load_vocab(hparams.tgt_vocab_path, tgt_field, pointer_copy_tokens=pointer_copy_tokens,
                   field_copy_tokens=field_copy_tokens)
        logger.info('[VOCAB] tgt vocab size: %d' % len(tgt_field.vocab.itos))
    else:
        logger.info('[VOCAB] Constructing a sharing vocab for the src and tgt')
        logger.info('[VOCAB] Loading src&tgt vocab from: %s' % hparams.vocab_path)
        load_vocab(hparams.vocab_path, tgt_field,
                   pointer_copy_tokens=pointer_copy_tokens,
                   field_copy_tokens=field_copy_tokens,
                   special_tokens=[src_field.unk_token, src_field.pad_token, src_field.init_token, src_field.eos_token])
        if 'src' in field_dict:
            src_field.vocab = tgt_field.vocab
            logger.info('[VOCAB] src vocab size: %d' % len(src_field.vocab.itos))
        else:
            logger.info('[VOCAB] src vocab size: -1')
        logger.info('[VOCAB] tgt vocab size: %d' % len(tgt_field.vocab.itos))

    if 'src' in field_dict:
        logger.info('[SRC_TAG_VOCAB] Constructing src pos tag vocab from %s' % hparams.src_tag_vocab_path)
        load_vocab(hparams.src_tag_vocab_path, src_tag_field)
    else:
        logger.info('[SRC_TAG_VOCAB] Ignored src pos tag vocab')
    tgt_bow_field.vocab = tgt_field.vocab

    if 'attribute_uni_key' in field_dict:
        if 'src' in field_dict:
            vocab = src_field.vocab
        else:
            vocab = tgt_field.vocab
        attribute_uni_word_field.vocab = vocab
        attribute_uni_key_field.vocab = vocab

        logger.info('[FIELD_KEY_VOCAB] Constructing field vocab from %s' % hparams.field_key_vocab_path)
        load_vocab(hparams.field_key_vocab_path, attribute_key_field)
        logger.info('[FIELD_KEY_VOCAB]  vocab size: %d' % len(attribute_key_field.vocab.itos))

        logger.info('[FIELD_WORD_VOCAB] Constructing field  word vocab from %s' % hparams.field_word_vocab_path)
        load_vocab(hparams.field_word_vocab_path, attribute_word_field)
        logger.info('[FIELD_WORD_VOCAB]  vocab size: %d' % len(attribute_word_field.vocab.itos))

        if hparams.field_word_tag_path != 'none':
            logger.info('[FIELD_WORD_TAG_VOCAB] Constructing tag word vocab from %s' % hparams.field_word_tag_path)
            load_vocab(hparams.field_word_tag_path, attribute_word_tag_field)
            logger.info('[FIELD_WORD_TAG_VOCAB]  vocab size: %d' % len(attribute_word_tag_field.vocab.itos))
        else:
            assert hparams.field_tag_usage == 'none', 'please add a field_tag vocab'
            field_dict['attribute_word_tag'] = None

        if hparams.field_key_tag_path != 'none':
            logger.info('[FIELD_KEY_TAG_VOCAB] Constructing tag key vocab from %s' % hparams.field_key_tag_path)
            load_vocab(hparams.field_key_tag_path, attribute_key_tag_field)
            logger.info('[FIELD_KEY_TAG_VOCAB]  vocab size: %d' % len(attribute_key_tag_field.vocab.itos))
        else:
            assert hparams.field_tag_usage in ['none', 'general'], 'please add a field_tag vocab'
            attribute_key_tag_field.vocab = attribute_word_tag_field.vocab
            field_dict['attribute_key_tag_field'] = None

    if hparams.task_mode != 'table2text':
        sort_key = lambda x: len(x.tgt) + len(x.src) * (hparams.max_src_len + 5)
    else:
        sort_key = lambda x: len(x.attribute_word) + len(x.tgt) * (hparams.max_src_len + 5)

    device = 'cuda' if hparams.cuda else 'cpu'

    # Update Fields:
    field_orders = []
    for field in fields:
        if field_dict[field[0]] is None:
            logger.info('[FIELD] Removing the invalid field %s:' % field[0])
            del field_dict[field[0]]
            assert field[0] not in field_dict
        else:
            logger.info('[FIELD] Keep the valid field %s:' % field[0])
            field_orders.append(field)

    val, max_val_len = get_dataset(hparams, hparams.val_data_path_prefix,
                                   field_orders=field_orders, field_dict=field_dict,
                                   random_order=hparams.random_train_field_order)

    test, max_test_len = get_dataset(hparams, hparams.test_data_path_prefix if not test_data_path else test_data_path,
                                     field_orders=field_orders, field_dict=field_dict,
                                     random_order=hparams.random_test_field_order)

    if hparams.copy:
        assert max_val_len + 1 < max_copy_token_num, max_val_len
        assert max_test_len + 1 < max_copy_token_num, max_test_len

    if not is_eval:
        logger.info('[DATASET] Training Mode')
        train, max_train_len = get_dataset(hparams, hparams.train_data_path_prefix,
                                           field_orders=field_orders, field_dict=field_dict,
                                           random_order=hparams.random_train_field_order,
                                           copy_dropout=hparams.align_dropout)
        if hparams.copy:
            assert max_train_len + 1 < max_copy_token_num, max_train_len
        train_iter = BucketIterator(train, batch_size=batch_size, repeat=False, shuffle=True,
                                    sort_key=sort_key, sort=False, train=True, sort_within_batch=True,
                                    device=device)
        val_iter = BucketIterator(val, batch_size=batch_size, repeat=False, shuffle=True,
                                  sort_key=sort_key, sort=False, train=False, sort_within_batch=True,
                                  device=device)
        test_iter = Iterator(test, batch_size=batch_size, repeat=False, shuffle=False,
                             sort_key=sort_key, sort=False, train=False, sort_within_batch=False,
                             device=device)
        return train_iter, val_iter, test_iter, field_dict
    else:
        logger.info('[DATASET] Eval/Inference Mode')
        val_iter = Iterator(val, batch_size=batch_size, repeat=False, shuffle=False,
                            sort_key=sort_key, sort=False, train=False, sort_within_batch=False,
                            device=device)
        test_iter = Iterator(test, batch_size=batch_size, repeat=False, shuffle=False,
                             sort_key=sort_key, sort=False, train=False, sort_within_batch=False,
                             device=device)
        return None, val_iter, test_iter, field_dict
