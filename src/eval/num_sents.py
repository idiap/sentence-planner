#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Computes number of summary sentences. """

import numpy as np
import pandas as pd

from interface import Evaluation


class NumberOfSentences(Evaluation):
    """ Counts the number of sentences in each summary. """

    def __init__(self, input_dir, sent_sep='<q>'):
        super().__init__(input_dir)
        self.name = 'num_sents'
        self.sent_sep = sent_sep

    def run(self):
        num_sents = []
        summaries = self.read_candidate_summaries()
        for summary in summaries:
            sents = summary.split(self.sent_sep)
            sents = list(filter(None, map(str.strip, sents)))  # remove empty sentences
            num_sents.append(len(sents))
        df = pd.DataFrame({'num_sents': np.mean(num_sents)}, index=[0])
        df.to_csv(self.get_output_path(), index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes number of summary sentences.')
    parser.add_argument('--eval_dir', required=True, help='Evaluation directory')
    args = parser.parse_args()
    NumberOfSentences(args.eval_dir).run()
