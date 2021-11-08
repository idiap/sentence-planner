#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Uses Google's ROUGE implementation to compute scores between candidate and reference summary. """

from rouge_score import rouge_scorer
from rouge_score.scoring import BootstrapAggregator


class RougeScores:

    def __init__(self, use_stemmer=True):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=use_stemmer)
        self.aggregator = BootstrapAggregator()

    def compute_scores(self, candidates_path, references_path, newline_token=None):
        # load summaries
        candidates = [line.strip() for line in open(candidates_path, encoding='utf-8')]
        references = [line.strip() for line in open(references_path, encoding='utf-8')]
        assert len(candidates) == len(references)

        # compute ROUGE scores for each candidate-reference pair
        for i, (c, r) in enumerate(zip(candidates, references)):
            if len(c) < 1 or len(r) < 1:
                print('Empty result in line %d - candidate len: %d, reference len: %d' % (i, len(c), len(r)))
                continue
            if newline_token:
                c = c.replace(newline_token, '\n')
                r = r.replace(newline_token, '\n')
            self.aggregator.add_scores(self.scorer.score(r, c))
        results = self.aggregator.aggregate()

        # use expected formatting
        output = {}
        for n in ('1', '2', 'Lsum'):
            out = 'l' if n == 'Lsum' else n
            output['rouge_%s_precision' % out] = results['rouge%s' % n].mid.precision
            output['rouge_%s_recall' % out] = results['rouge%s' % n].mid.recall
            output['rouge_%s_f_score' % out] = results['rouge%s' % n].mid.fmeasure
        return output


if __name__ == '__main__':
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Compute ROUGE scores with Google's Python implementation.")
    parser.add_argument('--candidates_path', required=True, help='Path to candidate summaries.')
    parser.add_argument('--references_path', required=True, help='Path to reference summaries.')
    parser.add_argument('--newline_token', default='<q>', help='Newline token in multi-line summaries.')
    args = parser.parse_args()

    np.random.seed(123)
    scores = RougeScores(use_stemmer=True)
    results = scores.compute_scores(args.candidates_path, args.references_path, args.newline_token)
    for k in sorted(results):
        print('%-20s %.5f' % (k, results[k]))
