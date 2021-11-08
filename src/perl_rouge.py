#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Computes ROUGE scores with the Perl script. Needs libxml-parser-perl installed. """

import os
from others.utils import test_rouge


def compute_scores(tmpdir, candidates_path, references_path):
    os.makedirs(tmpdir, exist_ok=True)  # temporary dir for Perl script's xml files and config
    return test_rouge(tmpdir, candidates_path, references_path, verbose=False)


if __name__ == '__main__':
    import logging
    import argparse

    parser = argparse.ArgumentParser(description='Compute ROUGE scores with Perl script.')
    parser.add_argument('--candidates_path', required=True, help='Path to candidate summaries.')
    parser.add_argument('--references_path', required=True, help='Path to reference summaries.')
    parser.add_argument('--tmpdir', default='/tmp/presumm', help='Path to tmp dir for Perl script')
    parser.add_argument('--csv', action='store_true', help='Output csv file instead of full results')
    args = parser.parse_args()

    logging.disable(logging.INFO)  # silence logging
    results = compute_scores(args.tmpdir, args.candidates_path, args.references_path)
    logging.disable(logging.NOTSET)
    if args.csv:
        print('rouge_1_f_score,rouge_2_f_score,rouge_l_f_score')
        print('%.2f,%.2f,%.2f' %
              (results['rouge_1_f_score'] * 100, results['rouge_2_f_score'] * 100, results['rouge_l_f_score'] * 100))
    else:
        for k in sorted(results):
            if not k.endswith('cb') and not k.endswith('ce'):
                print('%-20s %.5f' % (k, results[k]))
