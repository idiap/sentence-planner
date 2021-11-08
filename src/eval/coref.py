#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Computes number of coreference links across sentence boundaries. """

import numpy as np
import pandas as pd

from interface import Evaluation
from eval_utils import create_sentence_pairs


class Coref(Evaluation):
    """ Computes number of coreference links across sentence boundaries. """

    def __init__(self, input_dir, sent_sep='<q>', only_consecutive_sents=False):
        super().__init__(input_dir)
        self.name = 'coref'
        self.conda_env = 'coref'
        self.sent_sep = sent_sep
        self.only_consecutive_sents = only_consecutive_sents
        self.nlp = None

    def run(self):
        # read summaries
        summaries = self.read_candidate_summaries()

        # init coref pipeline
        self.init_coref()

        num_corefs = []
        for summary in summaries:
            if self.only_consecutive_sents:
                corefs = 0
                pairs = create_sentence_pairs(summary, self.sent_sep)
                for s1, s2 in pairs:
                    doc = self.nlp(s1 + ' ' + s2)
                    corefs += Coref._count_corefs(doc)
            else:
                doc = self.nlp(summary.replace(self.sent_sep, ' '))
                corefs = Coref._count_corefs(doc)

            # normalize by the number of sentences
            num_sents = summary.count(self.sent_sep) + 1
            num_corefs.append(corefs / num_sents)

        df = pd.DataFrame({'mean_corefs': np.mean(num_corefs)}, index=[0])
        df.to_csv(self.get_output_path(), index=False)

    def init_coref(self):
        """ Set up coreference spacy pipeline. """
        import spacy
        import neuralcoref
        self.nlp = spacy.load('en_core_web_lg')
        neuralcoref.add_to_pipe(self.nlp)

    @staticmethod
    def _count_corefs(doc):
        # count all coreferences across sentence boundaries
        num_corefs = 0
        for cluster in doc._.coref_clusters:
            # for each cluster, determine how many sentences are involved
            num_sents = len(set(map(lambda mention: mention.sent.start, cluster.mentions)))
            assert num_sents <= len(cluster)
            num_corefs += num_sents - 1
        return num_corefs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes number of coreference links across sentence boundaries.')
    parser.add_argument('--eval_dir', required=True, help='Evaluation directory')
    args = parser.parse_args()
    Coref(args.eval_dir).run()
