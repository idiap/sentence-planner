#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Computes the importance of the sentence representation in the SentSumm sentence conditioning model. """

import os
import time
import glob
import torch
import random
import argparse
import numpy as np
import pandas as pd
from transformers import BertTokenizer

from models.data_loader import load_dataset
from models.data_loader import Dataloader
from sentsumm.model import SentSumm
from train_abstractive import model_flags


def parse_args():
    parser = argparse.ArgumentParser(description='Outputs attention to sentence representation.')
    parser.add_argument('--bert_data_path', default='cnndm_bert/cnndm', help='Path to data')
    parser.add_argument('--model_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--result_path', required=True, help='Path to result folder')
    parser.add_argument('--pretrained_dir', default='bert-base-uncased', help='Pretrained Bert')
    parser.add_argument('--log_file', default='gxa.log', help='Path to log file')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--step', type=int, default=None, metavar='N', help='Use checkpoint of model step N')
    parser.add_argument('--num_examples', type=int, default=100, help='Stop after this many examples (0 for all)')
    parser.add_argument('--pick_scores', default='predicted', choices=['target', 'predicted'], help='Take gradient wrt. target or predicted scores.')
    parser.add_argument('--num_ig_steps', type=int, default=1, help='Number of steps to approximate integral in integrated gradients.')
    parser.add_argument('--baseline', default='zero', choices=['zero', 'mean'], help='Zero or mean vector baseline for integrated gradients.')

    # necessary architecture arguments for model initialization
    parser.add_argument('--model', default='sentsumm', choices=['presumm', 'sentsumm'], help='Model name')
    parser.add_argument('--pretrain_dec', action='store_true', help='Use pretrained Bert to initialize decoder')
    parser.add_argument('--train_first_sent_emb', action='store_true', help='Train first sentence representation')
    parser.add_argument('--use_dec_src_attn', action='store_true', help='Allow attention from word decoder to article')
    parser.add_argument('--use_sent_cond', action='store_true', help='Condition on sentence representation in decoder')
    parser.add_argument('--fix_sent_head', action='store_true', help='Fix a head to look at sentence representation in decoder')
    parser.add_argument('--no_sent_repr', action='store_true', help='Use only the article in the decoder attention and ignore the sentence representation.')
    parser.add_argument('--use_lm_only', action='store_true', help='Use only the language model to predict the summary, no encoder attention.')
    parser.add_argument('--see_past_words', action='store_true', help='Allow to see words of previous sentences when encoding summary sentences.')
    parser.add_argument('--see_past_sents', action='store_true', help='Allow to see sentence representations of previous sentences in the sentence planner.')
    parser.add_argument('--share_word_enc', action='store_true', help='Share the word encoder for article and summary inputs.')
    parser.add_argument('--sentplan_loss_weight', type=float, default=0.0, help='Weight for sentence planner loss (0 turns it off)')
    parser.add_argument('--do_finetune', action='store_true', help='Finetune model from `train_from` checkpoint')
    parser.add_argument('--load_partial', action='store_true', help="Load model from checkpoint even if it doesn't match exactly")
    parser.add_argument('--sentgen_layers', '--num_layers_sent_gen', type=int, default=2, help='Number of layers in sentence generator')
    parser.add_argument('--sentgen_heads', type=int, default=12, help='Number of attention heads in sentence generator')
    parser.add_argument('--sentgen_hidden_size', type=int, default=768, help='Hidden size in sentence generator')
    parser.add_argument('--sentgen_ff_size', type=int, default=3072, help='Feed-forward intermediate size in sentence generator')
    parser.add_argument('--sentgen_attn_dropout', type=float, default=0.1, help='Attention dropout in sentence generator')
    parser.add_argument('--sentgen_hidden_dropout', type=float, default=0, help='Embedding dropout in sentence generator')
    return parser.parse_known_args()[0]  # can ignore other args as they will be loaded from checkpoint


def compute_mean_tensors(args, model, data_iter, num_layers, num_tensors):
    print('Computing mean baseline tensors...')
    tensors = torch.zeros([args.num_examples, num_layers, num_tensors, args.dec_hidden_size])
    for n, batch in enumerate(data_iter):
        src = batch.src
        tgt = batch.tgt
        segs = batch.segs
        clss = batch.clss
        mask_src = batch.mask_src
        mask_tgt = batch.mask_tgt
        mask_cls = batch.mask_cls

        with torch.no_grad():
            (output_states, sent_cond_tensors), _ = model(src, tgt, segs, clss, mask_src, mask_tgt, mask_cls)
            for i in range(num_layers):
                for j in range(num_tensors):
                    tensors[n, i, j] = sent_cond_tensors[i][j].squeeze(0).mean(dim=0)

        if n + 1 >= args.num_examples:
            break

    return tensors.mean(dim=0)


def get_attention(args, cp):
    # load checkpoint
    checkpoint = torch.load(cp, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    # initialize random number generator
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]'],
               'CLS': tokenizer.vocab['[CLS]'], 'SEP': tokenizer.vocab['[SEP]']}

    # load test dataset
    dataset = load_dataset(args, 'valid', shuffle=False)

    # load model from checkpoint
    model = SentSumm(args, 'cpu', symbols, checkpoint=checkpoint)
    model.eval()
    valid_iter = Dataloader(args, dataset, batch_size=1, device='cpu', shuffle=False, is_test=False)

    # run validation
    num_layers = args.dec_layers
    stats = ['grad_times_act', 'grad_times_act_ig']
    num_stats = len(stats)
    num_tensors = 2  # attention output, sentence representation

    # define baseline tensor
    if args.baseline == 'zero':
        baseline = torch.zeros([num_layers, num_tensors, 1, args.dec_hidden_size])
    else:
        # args.baseline == 'mean'
        baseline = compute_mean_tensors(args, model, valid_iter, num_layers, num_tensors)
        baseline = baseline.unsqueeze(2)  # add dim for sequence length

        # reload dataset
        dataset = load_dataset(args, 'valid', shuffle=False)
        valid_iter = Dataloader(args, dataset, batch_size=1, device='cpu', shuffle=False, is_test=False)

    # define integration steps
    alphas = np.linspace(0, 1, args.num_ig_steps) if args.num_ig_steps > 1 else [1]

    num_examples = 0
    durations = []
    stats_tensor = torch.zeros(num_stats, num_layers)
    for batch in valid_iter:
        start_time = time.time()

        src = batch.src
        tgt = batch.tgt
        segs = batch.segs
        clss = batch.clss
        mask_src = batch.mask_src
        mask_tgt = batch.mask_tgt
        mask_cls = batch.mask_cls

        # prediction length: target sequence length - 1
        pred_len = tgt.size(1) - 1

        # expand baseline to seq_length
        baseline_batch = baseline.expand([-1, -1, pred_len, -1])

        # compute integrated gradients when moving from baseline input (0-vector) to true input
        grads_integrated = torch.zeros(num_tensors * num_layers, 1, pred_len, args.dec_hidden_size)
        for alpha in alphas:
            # sent_cond_tensors: attention output, sentence representation (scaled with alpha from baseline)
            (output_states, sent_cond_tensors), _ = model(src, tgt, segs, clss, mask_src, mask_tgt, mask_cls,
                                                          alpha=alpha, baseline=baseline_batch)

            # compute and select scores (either of predicted class or target class
            scores = model.generator(output_states.squeeze(0))
            if args.pick_scores == 'target':
                # pick true target's score
                targets = tgt[0, 1:]
                scores = [scores[i][t] for i, t in enumerate(targets)]
            else:
                # pick predicted score
                scores = scores.max(dim=1).values.unbind()

            # concat input tensors into list
            assert num_tensors == len(sent_cond_tensors[0]), 'Unexpected number of returned sent_cond_tensors'
            input_tensors = []
            for layer_tensors in sent_cond_tensors:
                input_tensors.extend(layer_tensors)
            assert len(input_tensors) == num_tensors * num_layers, 'Unexpected number of intermediate tensors'

            # compute gradients
            model.zero_grad()
            grads = torch.autograd.grad(scores, input_tensors)

            # sum up gradients for integrated gradients method
            grads_integrated += torch.stack(grads)

        # get statistics per layer
        for i in range(num_layers):
            # unpack gradients and tensors
            grad_hidden = grads[i * 2]
            grad_sent = grads[i * 2 + 1]
            grad_integrated_hidden = grads_integrated[i * 2]
            grad_integrated_sent = grads_integrated[i * 2 + 1]
            tensor_hidden = input_tensors[i * 2]
            tensor_sent = input_tensors[i * 2 + 1]
            baseline_hidden = baseline[i, 0].unsqueeze(0).expand_as(tensor_hidden)
            baseline_sent = baseline[i, 1].unsqueeze(0).expand_as(tensor_sent)

            # gradient x activations
            gxa_hidden = torch.norm(grad_hidden * tensor_hidden).item()
            gxa_sent = torch.norm(grad_sent * tensor_sent).item()
            stats_tensor[stats.index('grad_times_act'), i] += gxa_sent / (gxa_hidden + gxa_sent)

            # integrated gradient x activations
            gxa_integrated_hidden = torch.norm((tensor_hidden - baseline_hidden) * grad_integrated_hidden).item()
            gxa_integrated_sent = torch.norm((tensor_sent - baseline_sent) * grad_integrated_sent).item()
            stats_tensor[stats.index('grad_times_act_ig'), i] += gxa_integrated_sent / (gxa_integrated_hidden + gxa_integrated_sent)

        # report progress
        num_examples += 1
        duration = time.time() - start_time
        durations.append(duration)
        print('Processed %4d/%d, took %.1fs, mean %.1fs' % (num_examples, args.num_examples, duration, np.mean(durations)), flush=True)

        # stop when number of examples is reached
        if 0 < args.num_examples <= num_examples:
            break

    # normalize by the number of examples
    stats_tensor /= num_examples

    # save to df
    columns = ['layer'] + stats
    df = pd.DataFrame(columns=columns)
    for l in range(num_layers):
        results = {'layer': l}
        for i, stat in enumerate(stats):
            results[stat] = stats_tensor[i, l].item()
        df = df.append(results, ignore_index=True)
    df.to_csv(os.path.join(args.result_path, 'gxa-baseline_%s.csv' % args.baseline), index=False)


def main(args):
    fn = 'model_step_*.pt' if args.step is None else 'model_step_%d.pt' % args.step
    cp_files = glob.glob(os.path.join(args.model_path, fn))
    assert len(cp_files) > 0
    if len(cp_files) > 1:
        print('More than one model checkpoint available. Please provide the step to choose one.')
        return
    get_attention(args, cp_files[0])


if __name__ == '__main__':
    args = parse_args()
    args.output_attentions = False
    args.output_sent_cond = True
    args.visible_gpus = '-1'
    args.task = 'abs'
    args.max_tgt_len = 140
    args.max_pos = 512
    args.use_fixed_batch_size = True

    # sentsumm model args
    args.dec_layers = 6
    args.dec_hidden_size = 768
    args.dec_heads = 8
    args.dec_ff_size = 2048
    args.dec_dropout = 0.2
    args.finetune_enc = True
    args.tie_input_embs = False
    args.tie_decoder_embs = False

    main(args)
