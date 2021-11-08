#
# Original version by Yang Liu.
# Modifications by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
import pathlib
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs
from train_extractive import train_ext, validate_ext, test_ext

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='abs', type=str, choices=['ext', 'abs'])
    parser.add_argument("--encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("--mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("--bert_data_path", default='../bert_data_new/cnndm')
    parser.add_argument("--model_path", default='../models/')
    parser.add_argument("--result_path", default='../results/cnndm')
    parser.add_argument("--temp_dir", default='../temp')
    parser.add_argument("--pretrained_dir", default='bert-base-uncased')

    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--use_fixed_batch_size", type=str2bool, nargs='?', const=True, default=True, help='Use a tgt length-independent batch size.')

    parser.add_argument("--max_pos", default=512, type=int)
    parser.add_argument("--use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("--large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--load_from_extractive", default='', type=str)

    parser.add_argument("--sep_optim", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("--lr_bert", default=0.001, type=float)
    parser.add_argument("--lr_dec", default=0.02, type=float)
    parser.add_argument("--use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("--share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--dec_dropout", default=0.2, type=float)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dec_hidden_size", default=768, type=int)
    parser.add_argument("--dec_heads", default=8, type=int)
    parser.add_argument("--dec_ff_size", default=2048, type=int)
    parser.add_argument("--enc_hidden_size", default=512, type=int)
    parser.add_argument("--enc_ff_size", default=512, type=int)
    parser.add_argument("--enc_dropout", default=0.2, type=float)
    parser.add_argument("--enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("--ext_dropout", default=0.2, type=float)
    parser.add_argument("--ext_layers", default=2, type=int)
    parser.add_argument("--ext_hidden_size", default=768, type=int)
    parser.add_argument("--ext_heads", default=8, type=int)
    parser.add_argument("--ext_ff_size", default=2048, type=int)

    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--generator_shard_size", default=32, type=int)
    parser.add_argument("--alpha",  default=0.95, type=float)
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--min_length", default=50, type=int)
    parser.add_argument("--max_length", default=200, type=int)
    parser.add_argument("--max_tgt_len", default=140, type=int)



    parser.add_argument("--param_init", default=0, type=float)
    parser.add_argument("--param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("--optim", default='adam', type=str)
    parser.add_argument("--lr", default=1, type=float)
    parser.add_argument("--beta1", default= 0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--warmup_steps", default=8000, type=int)
    parser.add_argument("--warmup_steps_bert", default=2500, type=int)
    parser.add_argument("--warmup_steps_dec", default=2500, type=int)
    parser.add_argument("--max_grad_norm", default=0, type=float)

    parser.add_argument("--save_checkpoint_steps", default=10000, type=int)
    parser.add_argument("--num_checkpoints_to_keep", default=0, type=int)
    parser.add_argument("--accum_count", default=5, type=int)
    parser.add_argument("--report_every", default=50, type=int)
    parser.add_argument("--train_steps", default=40000, type=int)
    parser.add_argument("--recall_eval", type=str2bool, nargs='?',const=True,default=False)


    parser.add_argument("--visible_gpus", default='-1', type=str)
    parser.add_argument("--gpu_ranks", default='0', type=str)
    parser.add_argument("--log_file", default='../logs/cnndm.log')
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--test_from", default='')
    parser.add_argument("--test_start_from", default=-1, type=int)

    parser.add_argument("--train_from", default='')
    parser.add_argument("--report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("--block_trigram", type=str2bool, nargs='?', const=True, default=True)

    # SentSumm arguments
    parser.add_argument('--model', default='sentsumm', choices=['presumm', 'sentsumm'], help='Model to use')
    parser.add_argument('--finetune_enc', type=str2bool, nargs='?', const=True, default=True, help='Finetune Bert word encoder')
    parser.add_argument('--pretrain_dec', type=str2bool, nargs='?', const=True, default=False, help='Use pretrained Bert to initialize decoder')
    parser.add_argument('--tie_input_embs', type=str2bool, nargs='?', const=True, default=False, help='Tie word encoder/decoder input embeddings')
    parser.add_argument('--tie_decoder_embs', type=str2bool, nargs='?', const=True, default=False, help='Tie word decoder input/output embeddings')
    parser.add_argument('--train_first_sent_emb', type=str2bool, nargs='?', const=True, default=False, help='Train first sentence representation')
    parser.add_argument('--use_dec_src_attn', type=str2bool, nargs='?', const=True, default=True, help='Allow attention from word decoder to article')
    parser.add_argument('--use_sent_cond', type=str2bool, nargs='?', const=True, default=True, help='Condition on sentence representation in decoder')
    parser.add_argument('--fix_sent_head', type=str2bool, nargs='?', const=True, default=False, help='Fix a head to look at sentence representation in decoder')
    parser.add_argument('--no_sent_repr', type=str2bool, nargs='?', const=True, default=False, help='Use only the article in the decoder attention and ignore the sentence representation.')
    parser.add_argument('--use_lm_only', type=str2bool, nargs='?', const=True, default=False, help='Use only the language model to predict the summary, no encoder attention.')
    parser.add_argument('--see_past_words', type=str2bool, nargs='?', const=True, default=False, help='Allow to see words of previous sentences when encoding summary sentences.')
    parser.add_argument('--see_past_sents', type=str2bool, nargs='?', const=True, default=True, help='Allow to see sentence representations of previous sentences in the sentence planner.')
    parser.add_argument('--share_word_enc', type=str2bool, nargs='?', const=True, default=True, help='Share the word encoder for article and summary inputs.')
    parser.add_argument('--output_attentions', type=str2bool, nargs='?', const=True, default=False, help='Return word generator attention')
    parser.add_argument('--sentplan_loss_weight', type=float, default=1, help='Weight for sentence planner loss (0 turns it off)')
    parser.add_argument('--do_finetune', type=str2bool, nargs='?', const=True, default=False, help='Finetune model from `train_from` checkpoint')
    parser.add_argument('--load_partial', type=str2bool, nargs='?', const=True, default=False, help="Load model from checkpoint even if it doesn't match exactly")
    parser.add_argument('--sentgen_layers', '--num_layers_sent_gen', type=int, default=2, help='Number of layers in sentence generator')
    parser.add_argument('--sentgen_heads', type=int, default=12, help='Number of attention heads in sentence generator')
    parser.add_argument('--sentgen_hidden_size', type=int, default=768, help='Hidden size in sentence generator')
    parser.add_argument('--sentgen_ff_size', type=int, default=3072, help='Feed-forward intermediate size in sentence generator')
    parser.add_argument('--sentgen_attn_dropout', type=float, default=0.1, help='Attention dropout in sentence generator')
    parser.add_argument('--sentgen_hidden_dropout', type=float, default=0, help='Embedding dropout in sentence generator')

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.task == 'abs'):
        if (args.mode == 'train'):
            train_abs(args, device_id)
            result_dir = os.path.dirname(args.result_path)
            pathlib.Path(os.path.join(result_dir, 'train.finished')).touch()
        elif (args.mode == 'validate'):
            validate_abs(args, device_id)
            result_dir = os.path.dirname(args.result_path)
            pathlib.Path(os.path.join(result_dir, 'validate.finished')).touch()
        elif (args.mode == 'lead'):
            baseline(args, cal_lead=True)
        elif (args.mode == 'oracle'):
            baseline(args, cal_oracle=True)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(args, device_id, cp, step)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device_id, cp, step)

    elif (args.task == 'ext'):
        if (args.mode == 'train'):
            train_ext(args, device_id)
        elif (args.mode == 'validate'):
            validate_ext(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device_id, cp, step)
