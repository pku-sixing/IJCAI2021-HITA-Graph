from collections import defaultdict
import torch.nn as nn
import torch
import os

from table2seq import t2s_data_utils as data_utils
from utils.logger import logger


# 权重初始化，默认xavier


def adjust_learning_rate(optimizer, rate=0.5, min_value=None):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    assert len(optimizer.param_groups) == 1
    for param_group in optimizer.param_groups:
        new_lr = param_group['lr'] * rate
        if min_value is not None:
            new_lr = max(new_lr, min_value)
        logger.info('[LEARNING RATE] adjusting %f to %f ' % (param_group['lr'], new_lr))
        param_group['lr'] = new_lr
        return param_group['lr']


def name_a_generation(args, mode):
    if args.disable_unk_output:
        mask_unk = 'masked'
    else:
        mask_unk = 'std'
    try:
        if args.random_test_field_order:
            mask_unk += '_roder'
        if args.beam_length_penalize == 'avg':
            mask_unk += '_avg'
        if args.repeat_index > 0:
            mask_unk += '_%d' % args.repeat_index

    except:
        pass
    return '%s_B%d_D%.2f_%s' % (mode, args.beam_width, args.diverse_decoding, mask_unk)


def reset_learning_rate(optimizer, lr_rate):
    assert len(optimizer.param_groups) == 1
    for param_group in optimizer.param_groups:
        new_lr = lr_rate
        logger.info('[LEARNING RATE] adjusting %f to %f ' % (param_group['lr'], new_lr))
        param_group['lr'] = new_lr
        return param_group['lr']


def show_parameters(model):
    trainable_param_counter = defaultdict(float)
    logger.info('Trainable Parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            prefix = name.split('.')[0]
            trainable_param_counter[prefix] += param.nelement()
            logger.info('{}-{}-{}-{}'.format(name, param.shape, param.dtype, param.device))
    logger.info('-------------')
    trainable_sum = 0
    for key in trainable_param_counter.keys():
        logger.info('[PARAMS-COUNTING] #%s:%.2fM' % (key, trainable_param_counter[key] / 1e6))
        trainable_sum += trainable_param_counter[key]
    logger.info('[PARAMS-SUM] #%s:%.2fM' % ('Trainable', trainable_sum / 1e6))

    non_trainable_param_counter = defaultdict(float)
    logger.info('###########')
    logger.info('Non-Trainable Parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            prefix = name.split('.')[0]
            non_trainable_param_counter[prefix] += param.nelement()
            logger.info('{}-{}-{}-{}'.format(name, param.shape, param.dtype, param.device))
    logger.info('-------------')
    non_trainable_sum = 0
    for key in non_trainable_param_counter.keys():
        logger.info('[PARAMS-COUNTING] #%s:%.2fM' % (key, non_trainable_param_counter[key] / 1e6))
        non_trainable_sum += non_trainable_param_counter[key]
    logger.info('[PARAMS-SUM] #%s:%.2fM' % ('Non-Trainable', non_trainable_sum / 1e6))
    logger.info('-------------')
    logger.info('[PARAMS-SUM] #%s:%.2fM' % ('Total', (trainable_sum + non_trainable_sum) / 1e6))


def try_restore_model(model_path, model, optimizer, states, best_model):
    if best_model:
        model_path = os.path.join(model_path, 'best_model')
    else:
        model_path = os.path.join(model_path, 'check_point')
    if not os.path.isdir(model_path):
        logger.info('[CHECKPOINT] No checkpoint is found! Dir is not existed :%s' % model_path)
        return False
    files = os.listdir(model_path)
    files = sorted(files, reverse=False)
    for file in files:
        if file[-3:] == '.pt':
            model_name = '%s/%s' % (model_path, file)
            checkpoint = torch.load(model_name)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['model'])
            for key in checkpoint['states']:
                states[key] = checkpoint['states'][key]
            logger.info('[CHECKPOINT] Loaded params from  :%s' % model_name)
            return True
    logger.info('[CHECKPOINT] No checkpoint is found in :%s' % model_path)
    return False


def save_model(model_path, epoch, val_loss, model, optimizer, arguments, states, best_model=True, clear_history=True):
    if best_model:
        model_path = os.path.join(model_path, 'best_model')
    else:
        model_path = os.path.join(model_path, 'check_point')
    if not os.path.isdir(model_path):
        logger.info('[CHECKPOINT] Creating model file:%s' % model_path)
        os.makedirs(model_path)
    model_name = '%s/seq2seq_%d_%d.pt' % (model_path, epoch, int(val_loss * 100))

    arguments_dict = {}
    for key, value in vars(arguments).items():
        arguments_dict[key] = value

    model_state = {
        'arguments': arguments_dict,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'states': states
    }
    torch.save(model_state, model_name)

    logger.info('[CHECKPOINT] Model has been saved to :%s' % model_name)
    if clear_history:
        logger.info('[CHECKPOINT] Removing old checkpoints')
        files = os.listdir(model_path)
        for file in files:
            file_name = '%s/%s' % (model_path, file)
            if file_name != model_name:
                logger.info('[CHECKPOINT] Removing %s' % file_name)
                try:
                    os.remove(file_name)
                except Exception as e:
                    print(e)


def write_results(args, file_prefix, output_queries, output_responses, output_scores, output_generations, raw_fields=None):
    if raw_fields is None:
        raw_fields = [None] * len(output_queries)
    with open(file_prefix + '.top1.txt', 'w+', encoding='utf-8') as ftop1:
        if args.copy or args.field_copy:
            for field, query, generations in zip(raw_fields, output_queries, output_generations):
                rst_generation = data_utils.restore_field_copy_example(field, query, generations[0])
                ftop1.write('%s\n' % rst_generation)
        else:
            for query, generations in zip(output_queries, output_generations):
                if len(generations) < 1:
                    continue
                ftop1.write('%s\n' % generations[0])

    with open(file_prefix + '.topk.txt', 'w+', encoding='utf-8') as ftopk:
        if args.copy or args.field_copy:
            for field, query, generations in zip(raw_fields, output_queries, output_generations):
                for generation in generations:
                    rst_generation = data_utils.restore_field_copy_example(field, query, generation)
                    ftopk.write('%s\n' % rst_generation)
        else:
            for generations in output_generations:
                for generation in generations:
                    ftopk.write('%s\n' % generation)

    with open(file_prefix + '.detail.txt', 'w+', encoding='utf-8') as fout:
        idx = 0
        for field, query, generation_beam, response, score_beam in zip(raw_fields, output_queries, output_generations,
                                                                output_responses, output_scores):
            if args.copy or args.field_copy:
                fout.write('#%d\n' % idx)
                fout.write('Query:\t%s\n' % query)
                fout.write('Response:\t%s\n' % response)
                for generation, score in zip(generation_beam, score_beam):
                    fout.write('Generation:\t%.4f\t%s (%s)\n' % (
                        score, data_utils.restore_field_copy_example(field, query, generation), generation))
                idx += 1
            else:
                fout.write('#%d\n' % idx)
                fout.write('Query:\t%s\n' % query)
                fout.write('Response:\t%s\n' % response)
                for generation, score in zip(generation_beam, score_beam):
                    fout.write('Generation:\t%.4f\t%s\n' % (score, generation))
                idx += 1


def init_network(model, method='xavier'):
    """
    :param model:
    :param method:
    :param seed:
    :return:
    """
    logger.info('[INIT] Initializing parameters: %s' % method)
    for name, w in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            if method == 'uniform' or w.dim() == 1:
                nn.init.uniform_(w, -0.1, 0.1)
            elif method == 'xavier':
                nn.init.xavier_uniform_(w)
            elif method == 'xavier_normal':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(w)
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)


def sequence_mask_fn(lengths, maxlen=None, dtype=torch.bool, mask_first=False):
    """

    :param lengths: [seq_len]
    :param maxlen: [seq_len,max_len]
    :param dtype:
    :return:
    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device, requires_grad=False)
    matrix = torch.unsqueeze(lengths, dim=-1)
    # mask = row_vector < matrix
    mask = row_vector.lt(matrix)
    if mask_first:
        mask[:, 0:1] = False
    mask = mask.type(dtype)
    return mask

def select_time_first_sequence_embedding(inputs, index):
    """

    :param inputs: (src_len, batch_size, dim)
    :param dim0's index: (batch_size)
    :return:
    """
    src_len, batch_size, dim = inputs.shape
    inputs = inputs.view(src_len * batch_size, dim)
    index_with_offset = index * batch_size + torch.arange(0, batch_size, dtype=torch.long, device=index.device)
    outputs = inputs.index_select(dim=0, index=index_with_offset)
    return outputs


def create_beam(tensor, beam_width, batch_dim):
    """

    :param tensor:
    :param beam_width:
    :param batch_dim:
    :return:
    """
    if isinstance(tensor, tuple) or isinstance(tensor, list):
        return tuple([torch.repeat_interleave(x, beam_width, batch_dim) for x in tensor])
    else:
        if tensor is None:
            return tensor
        return torch.repeat_interleave(tensor, beam_width, batch_dim)
