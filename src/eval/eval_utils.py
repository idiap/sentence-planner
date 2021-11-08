#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Utils for evaluation. """

import nltk


def preprocess_article(article):
    """ Remove special symbols from the article, lowercase, return tokenized words. """
    article = article.replace(' ##', '')
    article = article.replace('[CLS]', '')
    article = article.replace('[SEP]', '')
    article = article.lower()
    article_words = nltk.word_tokenize(article)
    return article_words


def preprocess_summary(summary, separator):
    """ Replace special symbols, split sentences, lowercase, return list of tokenized sentences. """
    summary = summary.replace(' ##', '')
    summary = summary.replace('[CLS]', '')
    summary = summary.replace('[SEP]', '')
    summary_sents = summary.split(separator)
    summary_sents = [s.lower() for s in summary_sents]
    summary_sents_words = [nltk.word_tokenize(s) for s in summary_sents]
    return summary_sents_words


def compute_rouge(references, candidates, separator):
    """ Compute Rouge F1 scores between reference and candidate summaries. """
    from rouge_score import rouge_scorer
    from rouge_score.scoring import BootstrapAggregator

    assert len(candidates) == len(references)

    # initialize scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    aggregator = BootstrapAggregator()

    # compute ROUGE scores for each candidate-reference pair
    for i, (c, r) in enumerate(zip(candidates, references)):
        if len(c) < 1 or len(r) < 1:
            print('Empty result in line %d - candidate len: %d, reference len: %d' % (i, len(c), len(r)))
            continue
        if separator:
            c = c.replace(separator, '\n')
            r = r.replace(separator, '\n')
        aggregator.add_scores(scorer.score(r, c))
    results = aggregator.aggregate()
    return results['rouge1'].mid.fmeasure, results['rouge2'].mid.fmeasure, results['rougeLsum'].mid.fmeasure


def create_sentence_pairs(summary, sentence_separator):
    """ Create pairs of subsequent summary sentences. """
    s = summary.split(sentence_separator)
    return [(s[i], s[i + 1]) for i in range(len(s) - 1)]
