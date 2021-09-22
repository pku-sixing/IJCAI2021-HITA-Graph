import os
import math
import torch
import time
from torch import optim
from torch.nn import functional as F
from table2seq.t2s_opts import parse_arguments
from table2seq.t2s_model import InfoSeq2Seq
from table2seq import t2s_data_utils as data_utils
from utils.logger import logger, init_logger
from utils.evaluation.evaluation_helper import ScoreManager
from utils.evaluation.eval import std_eval_with_args
from utils import model_helper


def evaluate(hparams, model, val_iter, vocab_size, src_field, tgt_field):
    model.eval()
    pad = tgt_field.vocab.stoi['<pad>']
    total_loss = 0
    total_ppl = 0
    num_cases = 0
    unk = tgt_field.vocab.stoi['<unk>']
    with torch.no_grad():
        for b, batch in enumerate(val_iter):
            if not model.table2text_mode:
                src, len_src = batch.src
            else:
                src, len_src = None, None
            tgt, len_tgt = batch.tgt
            num_cases += tgt.size()[1]
            output, token_output, _, _, _ = model(batch,  mode='eval')
            if hparams.unk_learning == 'none':
                loss = F.nll_loss(output[1:].view(-1, vocab_size),
                                  tgt[1:].contiguous().view(-1),
                                  ignore_index=pad, reduction='none')
                loss = loss.view(tgt[1:].size())
                loss = loss.sum(dim=0) / (len_tgt - 1)
            elif hparams.unk_learning == 'penalize':
                logits = output[1:].view(-1, vocab_size)
                labels = tgt[1:].contiguous().view(-1)
                loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
                is_unk = labels == unk
                loss = torch.where(is_unk, loss / is_unk.sum(), loss).sum() / (labels != pad).sum() * tgt.size()[1]
            elif hparams.unk_learning == 'skip':
                logits = output[1:].view(-1, vocab_size)
                labels = tgt[1:].contiguous().view(-1)
                loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
                is_unk = labels == unk
                loss = torch.where(is_unk, loss * 1e-20, loss).sum() / ((labels != pad).sum() - is_unk.sum()) * tgt.size()[1]

            total_loss += loss.sum().item()
            total_ppl += torch.exp(loss).sum().item()
            if b < 1:
                #
                output, token_output, scores = model.greedy_search_decoding(hparams, batch)
                # output, token_output, scores = model.beam_search_decoding(hparams, src, len_src, beam_width=5)
                queries = data_utils.translate(src, src_field)
                generations = data_utils.translate(token_output, tgt_field)
                if queries is None:
                    queries = ['None'] * len(generations)
                responses = data_utils.translate(tgt, tgt_field)
                if hparams.field_copy:
                    field_words = data_utils.translate(batch.attribute_uni_word[0], tgt_field)
                else:
                    field_words = [None] * len(tgt)
                idx = 0
                scores = scores.sum(dim=0)
                assert len(generations) == len(scores)
                beam_width = len(generations) // len(responses)
                beam_generations = []
                beam_scores = []
                for beam_id in range(len(generations)):
                    beam_generations.append(generations[beam_id * beam_width: beam_id * beam_width + beam_width])
                    beam_scores.append(scores[beam_id * beam_width: beam_id * beam_width + beam_width])
                for field, query, generation_beam, response, score_beam in zip(field_words, queries, beam_generations, responses,
                                                                        beam_scores):
                    logger.info('#%d' % idx)
                    logger.info('Query:%s' % query)
                    if hparams.copy or hparams.field_copy:
                        rst_response = data_utils.restore_field_copy_example(field, query, response)
                        logger.info('Response:%s (%s)' % (rst_response, response))
                    else:
                        logger.info('Response:%s' % response)

                    for generation, score in zip(generation_beam, score_beam):
                        if hparams.copy or hparams.field_copy:
                            rst_generation = data_utils.restore_field_copy_example(field, query, generation)
                            logger.info('Generation-RST %.4f :%s (%s)' % (score, rst_generation, generation))
                        else:
                            logger.info('Generation %.4f :%s' % (score, generation))
                    if idx >= 5:
                        break
                    idx += 1
                print('###########')
        return total_loss / num_cases, total_ppl / num_cases


def inference(hparams, model, val_iter, vocab_size, src_field, tgt_field):
    model.eval()
    output_queries = []
    output_responses = []
    output_scores = []
    output_generations = []
    total_loss = 0
    total_ppl = 0
    num_cases = 0
    pad = tgt_field.vocab.stoi['<pad>']
    with torch.no_grad():
        for b, batch in enumerate(val_iter):
            if not model.table2text_mode:
                src, len_src = batch.src
            else:
                src, len_src = None, None
            tgt, len_tgt = batch.tgt
            num_cases += tgt.size()[1]
            if hparams.beam_width == 0:
                output, token_output, scores = model.greedy_search_decoding(hparams, batch)
                loss = scores[1:,:]
                loss = -loss.sum(dim=0) / (len_tgt - 1)
                total_loss += loss.sum().item()
                total_ppl += torch.exp(loss).sum().item()
            else:
                output, token_output, scores = model.beam_search_decoding(hparams, batch, beam_width=hparams.beam_width)

            queries = data_utils.translate(src, src_field)
            generations = data_utils.translate(token_output, tgt_field)
            if queries is None:
                queries = ['None'] * len(generations)
            responses = data_utils.translate(tgt, tgt_field)
            scores = scores.sum(dim=0)
            assert len(generations) == len(scores)
            beam_width = len(generations) // len(responses)
            beam_generations = []
            beam_scores = []
            for beam_id in range(len(queries)):
                beam_generations.append(generations[beam_id * beam_width: beam_id * beam_width + beam_width])
                beam_scores.append(scores[beam_id * beam_width: beam_id * beam_width + beam_width])

            output_queries += queries
            output_responses += responses
            output_scores += beam_scores
            output_generations += beam_generations

        return output_queries, output_responses, output_scores, output_generations, total_loss / num_cases, total_ppl / num_cases



def step_report(epoch, step, total_step, loss, coverage_loss, total_bow_loss, total_post_loss, time_diff, optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(str(param_group['lr']))
    lrs = ','.join(lrs)
    time_per_step = time_diff / step
    remaining_time = time_per_step * (total_step - step)
    # 显示最准确的Loss
    loss = loss - coverage_loss - total_bow_loss - coverage_loss-total_post_loss
    logger.info(
        "[Epoch=%d/Step=%d-%d][lrs:%s][loss:%5.2f,cvg:%.3f,bow:%.3f,pos:%.3f][ppl:%5.2f][step_time:%.2fs][remain:%.2fs]" %
        (epoch, step, total_step, lrs, loss, coverage_loss, total_bow_loss, total_post_loss*10000, math.exp(loss), time_per_step, remaining_time))


def train_epoch(hparams, epoch, model, optimizer, train_iter, vocab_size, src_field, tgt_field):
    model.train()
    total_loss = 0
    total_post_loss = 0
    total_coverage_loss = 0
    total_bow_loss = 0
    start_time = time.time()
    pad = tgt_field.vocab.stoi['<pad>']
    unk = tgt_field.vocab.stoi['<unk>']
    for b, batch in enumerate(train_iter):
        tgt, len_tgt = batch.tgt
        optimizer.zero_grad()
        output, _, coverage_loss, bow_loss, post_loss = model(batch)
        logits = output[1:].view(-1, vocab_size)
        labels = tgt[1:].contiguous().view(-1)
        if hparams.unk_learning == 'none':
            loss = F.nll_loss(logits, labels, ignore_index=pad)
        elif hparams.unk_learning == 'penalize':
            loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
            is_unk = labels == unk
            loss = torch.where(is_unk, loss / is_unk.sum(), loss).sum() / (labels != pad).sum()
        elif hparams.unk_learning == 'skip':
            loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
            is_unk = labels == unk
            loss = torch.where(is_unk, loss * 1e-20, loss).sum() / ((labels != pad).sum()- is_unk.sum())
        else:
            raise NotImplementedError()

        if hparams.copy_coverage > 0.0 and hparams.copy:
            loss_for_bp = coverage_loss.mean() * hparams.copy_coverage + loss
        else:
            loss_for_bp = loss

        if bow_loss is not None and hparams.bow_loss > 0.0:
            loss_for_bp += bow_loss * hparams.bow_loss
            total_bow_loss += bow_loss.item() * hparams.bow_loss

        if coverage_loss is not None and hparams.copy_coverage > 0.0 :
            total_coverage_loss += coverage_loss.mean().item()
            total_post_loss += coverage_loss.mean().item()

        if post_loss is not None:
            total_post_loss += post_loss.item()
            loss_for_bp += post_loss



        loss_for_bp.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if b % hparams.report_steps == 0 and b != 0:
            total_post_loss = total_post_loss / hparams.report_steps
            total_loss = total_loss / hparams.report_steps
            total_coverage_loss = total_coverage_loss / hparams.report_steps
            total_bow_loss = total_bow_loss / hparams.report_steps
            time_diff = time.time() - start_time
            step_report(epoch, b, len(train_iter), total_loss, total_coverage_loss, total_bow_loss, total_post_loss,
                        time_diff, optimizer)
            total_loss = 0
            total_coverage_loss = 0
            total_bow_loss = 0
            total_post_loss = 0

def set_dynamic_vocabs(args, fields):
    if 'sub_attribute_key' in fields:
        args.__setattr__('sub_field_vocab_size', len(fields['sub_attribute_key'].vocab))
    if 'sub_attribute_word' in fields:
        args.__setattr__('sub_field_word_vocab_size', len(fields['sub_attribute_word'].vocab))
    if 'sub_src' in fields:
        args.__setattr__('sub_src_vocab_size', len(fields['sub_src'].vocab))


def train(args):
    states = {
        'val_loss': [],
        'val_ppl': [],
        'lr': args.lr,
        'epoch': 0,
        'best_epoch': 0,
        'best_val_loss': -1,
    }
    if args.cuda:
        logger.info('[PARAM] Enabling CUDA')
        assert torch.cuda.is_available()

    logger.info("[DATASET] Preparing dataset...")
    train_iter, val_iter, test_iter, fields = data_utils.load_dataset(args)
    if 'src' in fields:
        src_vocab_size, tgt_vocab_size = len(fields.get('src', None).vocab), len(fields['tgt'].vocab)
    else:
        tgt_vocab_size = len(fields['tgt'].vocab)
    logger.info("[TRAIN]: #Batches=%d (#Cases:%d))" % (len(train_iter), len(train_iter.dataset)))
    logger.info("[VALIDATION]: #Batches=%d (#Cases:%d))" % (len(val_iter), len(val_iter.dataset)))
    logger.info("[TEST]: #Batches=%d (#Cases:%d))" % (len(test_iter), len(test_iter.dataset)))

    logger.info("[PARAM] Setting vocab sizes")
    if args.copy:
        vocab_offset = args.max_copy_token_num
    else:
        vocab_offset = 0

    if args.field_copy:
        vocab_offset += args.max_kw_pairs_num


    args.__setattr__('tgt_vocab_size', tgt_vocab_size - vocab_offset)
    if 'attribute_key' in fields:
        args.__setattr__('field_vocab_size', len(fields['attribute_key'].vocab))
    if 'src' in fields:
        args.__setattr__('src_vocab_size', src_vocab_size - vocab_offset)
        args.__setattr__('src_tag_vocab_size', len(fields['src_tag'].vocab))
    if 'attribute_word' in fields:
        args.__setattr__('field_word_vocab_size', len(fields['attribute_word'].vocab))
    if 'attribute_word' in fields and args.field_tag_usage != 'none':
        args.__setattr__('field_pos_tag_vocab_size', len(fields['attribute_word_tag'].vocab))
    args.__setattr__('tgt_vocab_size_with_offsets', tgt_vocab_size)
    set_dynamic_vocabs(args, fields)
    logger.info('[VOCAB] tgt_vocab_size_with_offsets = ' + str(tgt_vocab_size))


    logger.info("[MODEL] Preparing model...")
    seq2seq = InfoSeq2Seq.build_s2s_model(args, fields.get('src', None), fields['tgt'])

    if args.cuda:
        seq2seq.to('cuda')

    model_helper.show_parameters(seq2seq)
    if args.init_lr > 0:
        logger.info("[LR] Using init LR %f" % args.init_lr)
        optimizer = optim.Adam(seq2seq.parameters(), lr=args.init_lr)
        states['lr'] = args.init_lr
    else:
        optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
        states['lr'] = args.lr
    logger.info(seq2seq)

    # Load model
    is_loaded = model_helper.try_restore_model(args.model_path, seq2seq, optimizer, states, best_model=False)
    if not is_loaded:
        logger.info("[PARAM] Using fresh params")
        model_helper.init_network(seq2seq, args.init)
        if args.init_word_vecs is True:
            assert args.pre_embed_dim == args.embed_size
            logger.info("[EMBED] Loading the pre-trained word_embeddings")
            # 使用预训练的Embedding
            assert args.embed_size == args.field_key_embed_size
            if 'attribute_key' in fields:
                data_utils.load_pretrain_embeddings(seq2seq.field_encoder.field_key_embed, fields['attribute_key'],
                                                    args.pre_embed_file, args.embed_size, char2word='avg',
                                                    suffix=args.dataset_version + 'attribute_key')
                if args.field_word_vocab_path != 'none':
                    data_utils.load_pretrain_embeddings(seq2seq.field_encoder.field_word_embedding,
                                                        fields['attribute_word'],
                                                        args.pre_embed_file, args.embed_size, char2word='avg',
                                                        suffix=args.dataset_version + 'attribute_word')
                if 'sub_attribute_key' in fields:
                    data_utils.load_pretrain_embeddings(seq2seq.field_encoder.sub_field_key_char_encoder.sub_embed,
                                                        fields.get('sub_attribute_key', None),
                                                        args.pre_embed_file, args.embed_size,
                                                        suffix=args.dataset_version + 'sub_attribute_key')
                if 'sub_attribute_word' in fields:
                    data_utils.load_pretrain_embeddings(seq2seq.field_encoder.sub_field_word_char_encoder.sub_embed,
                                                        fields.get('sub_attribute_word', None),
                                                        args.pre_embed_file, args.embed_size,
                                                        suffix=args.dataset_version + 'sub_attribute_word')
            if 'src' in fields:
                data_utils.load_pretrain_embeddings(seq2seq.src_embed, fields.get('src', None),
                                                    args.pre_embed_file, args.embed_size,
                                                    suffix=args.dataset_version+'src')
            if 'sub_src' in fields:
                data_utils.load_pretrain_embeddings(seq2seq.sub_src_embed, fields.get('sub_src', None),
                                                    args.pre_embed_file, args.embed_size,
                                                    suffix=args.dataset_version+'subsrc')
            if not args.share_embedding or 'src' not in fields:
                data_utils.load_pretrain_embeddings(seq2seq.dec_embed, fields['tgt'], args.pre_embed_file,
                                                    args.embed_size, suffix=args.dataset_version+'tgt')


    start_epoch = states['epoch']
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if epoch != start_epoch + 1 and (args.align_dropout > 0.0 and (args.copy or args.field_copy)):
            logger.info("[Dataset] Reloading & Aligning the dataset")
            train_iter, val_iter, test_iter, fields = data_utils.load_dataset(args)
        if epoch - states['best_epoch'] > 2:
            logger.info('[STOP] Early Stopped !')
            break
        start_time = time.time()
        num_batches = len(train_iter)
        logger.info('[NEW EPOCH] %d/%d, num of batches : %d' % (epoch, args.epochs, num_batches))
        train_epoch(args, epoch, seq2seq, optimizer, train_iter,
                    tgt_vocab_size, fields.get('src', None), fields['tgt'])
        val_loss, val_ppl_macro = evaluate(args, seq2seq, val_iter, tgt_vocab_size, fields.get('src', None), fields['tgt'])
        val_ppl = math.exp(val_loss)
        test_loss, test_ppl_macro = evaluate(args, seq2seq, test_iter, tgt_vocab_size, fields.get('src', None), fields['tgt'])
        test_ppl = math.exp(test_loss)

        logger.info("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2f/%5.2f | test_loss:%5.3f | test_pp:%5.2f/%5.2f"
                    % (epoch, val_loss, val_ppl, val_ppl_macro, test_loss, test_ppl, test_ppl_macro))
        time_diff = (time.time() - start_time) / 3600.0
        logger.info("[Epoch:%d] epoch time:%.2fH, est. remaining  time:%.2fH" %
                    (epoch, time_diff, time_diff * (args.epochs - epoch)))

        # Adjusting learning rate
        states['val_ppl'].append(val_ppl)
        states['val_loss'].append(val_loss)
        states['epoch'] = epoch
        if len(states['val_ppl']) >= 2:
            logger.info('[TRAINING] last->now valid ppl : %.3f->%.3f' % (states['val_ppl'][-2], states['val_ppl'][-1]))

        if len(states['val_ppl']) >= 2 and states['val_ppl'][-1] >= states['val_ppl'][-2]:
            logger.info('[TRAINING] Adjusting learning rate due to the increment of val_loss')
            new_lr = model_helper.adjust_learning_rate(optimizer, rate=args.lr_decay_rate)
            states['lr'] = new_lr
        else:
            if args.init_lr > 0 and epoch >= args.init_lr_decay_epoch - 1 and states['lr'] > args.lr:
                logger.info('[TRAINING] Try to decay the init decay, from epoch %d, and the next epoch is %d' %
                            (args.init_lr_decay_epoch, epoch+1))
                new_lr = model_helper.adjust_learning_rate(optimizer, rate=args.lr_decay_rate, min_value=args.lr)
                states['lr'] = new_lr

        # Save the model if the validation loss is the best we've seen so far.
        if states['best_val_loss'] == -1 or val_ppl < states['best_val_loss']:
            logger.info('[CHECKPOINT] New best valid ppl : %.3f->%.2f' % (states['best_val_loss'], val_ppl))
            model_helper.save_model(args.model_path, epoch, val_ppl, seq2seq, optimizer, args, states, best_model=True,
                       clear_history=True)
            states['best_val_loss'] = val_ppl
            states['best_epoch'] = epoch

        # Saving standard model
        model_helper.save_model(args.model_path, epoch, val_ppl, seq2seq, optimizer, args, states, best_model=False,
                   clear_history=True)


def eval(args):
    states = {}
    score_manager = ScoreManager(args.model_path, 'evaluation_score')
    if args.cuda:
        logger.info('[PARAM] Enabling CUDA')
        assert torch.cuda.is_available()

    logger.info("[DATASET] Preparing dataset...")
    train_iter, val_iter, test_iter, fields= data_utils.load_dataset(args, is_eval=True)
    if 'src' in fields:
        src_vocab_size, tgt_vocab_size = len(fields.get('src', None).vocab), len(fields['tgt'].vocab)
    else:
        tgt_vocab_size = len(fields['tgt'].vocab)
    logger.info("[VALIDATION]: #Batches=%d (#Cases:%d))" % (len(val_iter), len(val_iter.dataset)))
    logger.info("[TEST]: #Batches=%d (#Cases:%d))" % (len(test_iter), len(test_iter.dataset)))

    logger.info("[PARAM] Setting vocab sizes")
    if args.copy:
        vocab_offset = args.max_copy_token_num
    else:
        vocab_offset = 0

    if args.field_copy:
        vocab_offset += args.max_kw_pairs_num
    if 'src' in fields:
        args.__setattr__('src_vocab_size', src_vocab_size - vocab_offset)
        args.__setattr__('src_tag_vocab_size', len(fields['src_tag'].vocab))
    args.__setattr__('tgt_vocab_size', tgt_vocab_size - vocab_offset)
    args.__setattr__('tgt_vocab_size_with_offsets', tgt_vocab_size)
    if 'attribute_key' in fields:
        args.__setattr__('field_vocab_size', len(fields['attribute_key'].vocab))
        args.__setattr__('field_word_vocab_size', len(fields['attribute_word'].vocab))
        if args.field_tag_usage != 'none':
            args.__setattr__('field_pos_tag_vocab_size', len(fields['attribute_word_tag'].vocab))
    set_dynamic_vocabs(args, fields)
    logger.info("[MODEL] Preparing model...")
    seq2seq = InfoSeq2Seq.build_s2s_model(args, fields.get('src', None), fields['tgt'])
    model_helper.show_parameters(seq2seq)
    if args.cuda:
        seq2seq = seq2seq.to('cuda')
    logger.info(seq2seq)

    # Load model
    is_loaded = model_helper.try_restore_model(args.model_path, seq2seq, None, states, best_model=args.use_best_model)
    if not is_loaded:
        logger.info("[PARAM] Could not load a trained model！")
        return
    else:
        val_loss, val_ppl_macro = evaluate(args, seq2seq, val_iter, tgt_vocab_size, fields.get('src', None), fields['tgt'])
        val_ppl = math.exp(val_loss)
        test_loss, test_ppl_macro = evaluate(args, seq2seq, test_iter, tgt_vocab_size, fields.get('src', None), fields['tgt'])
        test_ppl = math.exp(test_loss)
        logger.info("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2f/%5.2f | test_loss:%5.3f | test_pp:%5.2f/%5.2f"
                    % (states['epoch'], val_loss, val_ppl, val_ppl_macro, test_loss, test_ppl, test_ppl_macro))

        score_manager.update('eval_val_loss', val_loss)
        score_manager.update('eval_val_ppl', val_ppl)
        score_manager.update('eval_val_macro_ppl', val_ppl_macro)
        score_manager.update('eval_test_loss', test_loss)
        score_manager.update('eval_test_ppl', test_ppl)
        score_manager.update('eval_test_macro_ppl', test_ppl_macro)


def infer(args):
    states = {}
    score_manager = ScoreManager(args.model_path, 'evaluation_score')

    if args.cuda:
        logger.info('[PARAM] Enabling CUDA')
        assert torch.cuda.is_available()

    logger.info("[DATASET] Preparing dataset...")
    train_iter, val_iter, test_iter, fields = data_utils.load_dataset(args, is_eval=True)
    if 'src' in fields:
        src_vocab_size, tgt_vocab_size = len(fields.get('src', None).vocab), len(fields['tgt'].vocab)
    else:
        tgt_vocab_size = len(fields['tgt'].vocab)
    logger.info("[VALIDATION]: #Batches=%d (#Cases:%d))" % (len(val_iter), len(val_iter.dataset)))
    logger.info("[TEST]: #Batches=%d (#Cases:%d))" % (len(test_iter), len(test_iter.dataset)))

    logger.info("[PARAM] Setting vocab sizes")
    if args.copy:
        vocab_offset = args.max_copy_token_num
    else:
        vocab_offset = 0

    if args.field_copy:
        vocab_offset += args.max_kw_pairs_num

    if 'src' in fields:
        args.__setattr__('src_vocab_size', src_vocab_size - vocab_offset)
        args.__setattr__('src_tag_vocab_size', len(fields['src_tag'].vocab))
    args.__setattr__('tgt_vocab_size', tgt_vocab_size - vocab_offset)
    args.__setattr__('tgt_vocab_size_with_offsets', tgt_vocab_size)
    if 'attribute_key' in fields:
        args.__setattr__('field_vocab_size', len(fields['attribute_key'].vocab))
        args.__setattr__('field_word_vocab_size', len(fields['attribute_word'].vocab))
        if args.field_tag_usage != 'none':
            args.__setattr__('field_pos_tag_vocab_size', len(fields['attribute_word_tag'].vocab))
    set_dynamic_vocabs(args, fields)

    generation_name = model_helper.name_a_generation(args, 'test')
    file_prefix = os.path.join(args.model_path, generation_name)

    if not args.skip_infer:
        logger.info("[MODEL] Preparing model...")
        seq2seq = InfoSeq2Seq.build_s2s_model(args, fields.get('src', None), fields['tgt'])
        model_helper.show_parameters(seq2seq)
        if args.cuda:
            seq2seq = seq2seq.to('cuda')
        logger.info(seq2seq)

        # Load model
        is_loaded = model_helper.try_restore_model(args.model_path, seq2seq, None, states, best_model=args.use_best_model)

        if not is_loaded:
            logger.info("[PARAM] Could not load a trained model！")
            return

        # Test Set
        if args.field_copy:
            raw_queries, raw_responses, raw_fields = data_utils.load_examples(args.test_data_path_prefix, field_words=True)
        else:
            raw_queries, raw_responses = data_utils.load_examples(args.test_data_path_prefix)
            raw_fields = [None] * len(raw_queries)
        output_queries, output_responses, output_scores, output_generations, test_loss, test_ppl_macro  = \
            inference(args, seq2seq, test_iter, tgt_vocab_size, fields.get('src', None), fields['tgt'])
        model_helper.write_results(args, file_prefix, raw_queries, raw_responses, output_scores, output_generations,
                                   raw_fields=raw_fields)

    logger.info('[STD Evaluation] Evaluating the test generations...')
    res_dict = std_eval_with_args(args, file_prefix, 'test')
    score_manager.update_group(generation_name, res_dict)

    if args.beam_width == 0 and args.skip_infer is False:
        test_ppl = math.exp(test_loss)
        score_manager.update('infer_b%d_test_loss' % args.beam_width, test_loss)
        score_manager.update('infer_b%d_test_ppl' % args.beam_width, test_ppl)
        score_manager.update('infer_b%d_test_macro_ppl' % args.beam_width, test_ppl_macro)


if __name__ == "__main__":
    try:
        init_logger()
        args = parse_arguments()
        random_seed = 12345
        torch.manual_seed(random_seed)
        if args.mode == 'train':
            train(args)
            eval(args)
        elif args.mode == 'eval':
            eval(args)
        elif args.mode == 'infer':
            infer(args)
    except KeyboardInterrupt as e:
        print("[STOP]", e)
