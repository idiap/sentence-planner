#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Counts the number of words in summary sentences. """

import numpy as np
import pandas as pd

from interface import Evaluation
from eval_utils import preprocess_summary


class SentenceLengths(Evaluation):
    """ Counts the number of words in summary sentences. """

    def __init__(self, input_dir, sent_sep='<q>'):
        super().__init__(input_dir)
        self.name = 'sent_len'
        self.sent_sep = sent_sep

    def run(self):
        sent_lens = []
        summaries = self.read_candidate_summaries()
        for summary in summaries:
            summary_sents_words = preprocess_summary(summary, self.sent_sep)
            sent_lens.extend([len(sent) for sent in summary_sents_words])
        df = pd.DataFrame({'sent_len': np.mean(sent_lens)}, index=[0])
        df.to_csv(self.get_output_path(), index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Counts the number of words in summary sentences.')
    parser.add_argument('--eval_dir', required=True, help='Evaluation directory')
    args = parser.parse_args()
    SentenceLengths(args.eval_dir).run()
