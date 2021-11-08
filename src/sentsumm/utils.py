#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Utils for SentSumm model. """

import torch


def parse_sentence_indices(cls, sep):
    """ Makes valid (non-overlapping, increasing) pairs of CLS and SEP tokens. """
    cls_clean, sep_clean = [], []
    i = 0
    while len(sep) > 0 and len(cls) > 0 and i < max(sep):
        cls_idx = cls.pop(0)
        while i > cls_idx:
            if len(cls) == 0:
                return cls_clean, sep_clean
            cls_idx = cls.pop(0)
        i = cls_idx
        sep_idx = sep.pop(0)
        while i > sep_idx:
            if len(sep) == 0:
                return cls_clean, sep_clean
            sep_idx = sep.pop(0)
        i = sep_idx
        cls_clean.append(cls_idx)
        sep_clean.append(sep_idx)
    return cls_clean, sep_clean


def parse_sentence_indices_test():
    assert parse_sentence_indices([1, 3, 5], [2, 4, 6]) == ([1, 3, 5], [2, 4, 6])
    assert parse_sentence_indices([0, 4], [2, 3, 6]) == ([0, 4], [2, 6])
    assert parse_sentence_indices([0, 4, 7], [3, 6]) == ([0, 4], [3, 6])
    assert parse_sentence_indices([0, 4, 7], [8]) == ([0], [8])
    assert parse_sentence_indices([0, 5, 9], [8, 19]) == ([0, 9], [8, 19])
    assert parse_sentence_indices([0, 5, 19], [8, 9]) == ([0], [8])
    assert parse_sentence_indices([0, 5, 7], [8, 9]) == ([0], [8])


def find_sentence_boundaries(symbols, input_ids):
    """ Finds the sentence boundaries in summary sentences. """
    assert input_ids.dim() == 2
    starts, ends, masks = [], [], []
    for ids in input_ids.tolist():
        end = []
        for i, id in enumerate(ids):
            if id == symbols['EOS'] or id == symbols['EOQ']:
                end.append(i)
        ends.append(end)
        starts.append([0] + list(map(lambda x: x + 1, end[:-1])))

    # pad to same length
    starts = pad(starts, symbols['PAD'])
    ends = pad(ends, symbols['PAD'])

    # mask is 1 where end is != 0
    for end in ends:
        mask = []
        for idx in end:
            mask.append(1 if idx != symbols['PAD'] else 0)
        masks.append(mask)
    return starts, ends, masks

def pad(data, pad_id):
    """ Pad all lists in data to the same length. """
    width = max(len(d) for d in data)
    return [d + [pad_id] * (width - len(d)) for d in data]


def find_sentence_boundaries_test():
    symbols = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'EOQ': 3}
    input_ids = torch.tensor([[1, 4, 2, 4, 4, 3], [1, 4, 4, 3, 0, 0]])
    starts, ends, masks = find_sentence_boundaries(symbols, input_ids)
    assert starts == [[0, 3], [0, 0]]
    assert ends == [[2, 5], [3, 0]]
    assert masks == [[1, 1], [1, 0]]


if __name__ == '__main__':
    parse_sentence_indices_test()
    find_sentence_boundaries_test()
