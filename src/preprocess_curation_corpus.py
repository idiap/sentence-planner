#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Preprocess the Curation corpus to use with PreSumm. """

import os
import nltk
import torch
import argparse
import unicodedata
import pandas as pd

from prepro.data_builder import BertData

MIN_SRC_NSENTS = 3
MAX_SRC_NSENTS = 100
MIN_SRC_NTOKENS_PER_SENT = 5
MAX_SRC_NTOKENS_PER_SENT = 200
MIN_TGT_NTOKENS = 5
MAX_TGT_NTOKENS = 500


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Curation corpus.')
    parser.add_argument('--article_path', default='curation-corpus-articles.csv', help='Path to articles')
    parser.add_argument('--summary_path', default='curation-corpus-base.csv', help='Path to summaries')
    parser.add_argument('--output_dir', default='curation_corpus', help='Path to output Bert pytorch files')
    parser.add_argument('--splits', type=int, nargs=3, default=[80, 10, 10], help='Train/valid/test splits (3 ints)')
    parser.add_argument('--chunks', type=int, default=4000, help='Size of dataset chunks')
    parser.add_argument('--do_filter', action='store_true', help='Filter articles and summaries for number of tokens/sentences.')
    return parser.parse_args()


def split_into_sents(text):
    """ Splits the input `text` into sentences. """
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())  # replace multiple consecutive whitespace
    sents = nltk.sent_tokenize(text)
    return sents


def create_dataset(output_dir, formatter, articles, summaries, name, chunk_size, do_filter):
    """ Create a dataset with the given name. """
    dataset = []
    chunk_counter = 0
    total_num_examples = 0
    for article, summary in zip(articles, summaries):
        if pd.isna(article) or pd.isna(summary):
            continue
        sents_article = split_into_sents(article)
        summary = unicodedata.normalize('NFKC', summary)  # remove non-breaking spaces and possibly others
        sents_summary = split_into_sents(summary)
        data = formatter.preprocess_curation_corpus(sents_article, sents_summary, do_filter=do_filter)
        if data is None:
            continue
        data_dict = {
            'src': data[0],
            'segs': data[1],
            'tgt': data[2],
            'src_txt': data[3],
            'tgt_txt': data[4],
            'src_sent_labels': [],
            'clss': [],
        }
        dataset.append(data_dict)
        if len(dataset) >= chunk_size:
            dataset_path = os.path.join(output_dir, 'curation.%s.%d.bert.pt' % (name, chunk_counter))
            print('Saving chunk with %d examples to %s' % (len(dataset), dataset_path))
            torch.save(dataset, dataset_path)
            total_num_examples += len(dataset)
            dataset = []
            chunk_counter += 1
    if len(dataset) > 0:
        dataset_path = os.path.join(output_dir, 'curation.%s.%d.bert.pt' % (name, chunk_counter))
        print('Saving chunk with %d examples to %s' % (len(dataset), dataset_path))
        torch.save(dataset, dataset_path)
        total_num_examples += len(dataset)
    print('Saved %d examples for dataset %s' % (total_num_examples, name))


def main(args):
    articles_df = pd.read_csv(args.article_path)
    summaries_df = pd.read_csv(args.summary_path)

    # check indices match
    assert len(articles_df) == len(summaries_df)
    indices_match = articles_df['url'] == summaries_df['url']
    assert len(articles_df) == indices_match.sum()

    # compute split sizes
    assert args.splits[0] + args.splits[1] + args.splits[2] == 100
    total_size = len(articles_df)
    train_size = int(total_size * args.splits[0] / 100)
    valid_size = int(total_size * args.splits[1] / 100)
    test_size = total_size - train_size - valid_size
    print('Creating splits of size: %d/%d/%d' % (train_size, valid_size, test_size))

    # initialize formatter
    formatter = BertData(argparse.Namespace(**{
        'min_src_nsents': MIN_SRC_NSENTS,
        'max_src_nsents': MAX_SRC_NSENTS,
        'min_src_ntokens_per_sent': MIN_SRC_NTOKENS_PER_SENT,
        'max_src_ntokens_per_sent': MAX_SRC_NTOKENS_PER_SENT,
        'min_tgt_ntokens': MIN_TGT_NTOKENS,
        'max_tgt_ntokens': MAX_TGT_NTOKENS,
    }), use_huggingface_tokenizer=True)

    # create splits
    create_dataset(
        args.output_dir,
        formatter,
        articles_df['article_content'][0:train_size],
        summaries_df['summary'][0:train_size],
        name='train',
        chunk_size=args.chunks,
        do_filter=args.do_filter,
    )
    create_dataset(
        args.output_dir,
        formatter,
        articles_df['article_content'][train_size:train_size + valid_size],
        summaries_df['summary'][train_size:train_size + valid_size],
        name='valid',
        chunk_size=args.chunks,
        do_filter=args.do_filter,
    )
    create_dataset(
        args.output_dir,
        formatter,
        articles_df['article_content'][-test_size:],
        summaries_df['summary'][-test_size:],
        name='test',
        chunk_size=args.chunks,
        do_filter=args.do_filter,
    )


if __name__ == '__main__':
    main(parse_args())
