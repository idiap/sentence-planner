#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Computes the proportion of novel bigrams in the summary. """

import numpy as np
import pandas as pd

from interface import Evaluation
from eval_utils import preprocess_article, preprocess_summary


class NovelBigrams(Evaluation):

    def __init__(self, input_dir, sent_sep='<q>'):
        super().__init__(input_dir)
        self.name = 'novel_bigrams'
        self.sent_sep = sent_sep

    def run(self):
        # read articles and summaries
        articles = self.read_articles()
        summaries = self.read_candidate_summaries()
        assert len(articles) == len(summaries)

        # compute novel bigrams for each article-summary pair
        novel_bigrams = []
        for article, summary in zip(articles, summaries):
            article_words = preprocess_article(article)
            summary_tokenized_sents = preprocess_summary(summary, self.sent_sep)
            novel_bigrams.append(NovelBigrams.compute_novel_bigrams(article_words, summary_tokenized_sents))
        novel_bigrams = [score for score in novel_bigrams if score is not None]  # filter bad summaries

        # write results
        df = pd.DataFrame({'novel_bigrams': np.mean(novel_bigrams)}, index=[0])
        df.to_csv(self.get_output_path(), index=False)

    @staticmethod
    def compute_novel_bigrams(article_words, summary_tokenized_sents):
        """ Computes the proportion of novel bigrams in the summary. """
        bigrams_article = set((article_words[i], article_words[i + 1]) for i in range(len(article_words) - 1))
        bigrams_summary = set()
        for sentence_words in summary_tokenized_sents:
            bigrams_summary |= set((sentence_words[i], sentence_words[i + 1]) for i in range(len(sentence_words) - 1))
        return len(bigrams_summary - bigrams_article) / len(bigrams_summary) if len(bigrams_summary) > 0 else None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes the proportion of novel bigrams in the summary.')
    parser.add_argument('--eval_dir', required=True, help='Evaluation directory')
    args = parser.parse_args()
    NovelBigrams(args.eval_dir).run()
