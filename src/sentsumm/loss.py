#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Sentence planner loss. """

from torch.nn.functional import mse_loss

from models.loss import NMTLossCompute, shards
from models.reporter import Statistics


def sentsumm_loss(generator, symbols, vocab_size, device, train=True, label_smoothing=0.0, vocab=None, sentplan_loss_weight=0.0):
    compute = SentencePlannerLoss(generator, symbols, vocab_size, label_smoothing=label_smoothing if train else 0.0,
                                  vocab=vocab, sentplan_loss_weight=sentplan_loss_weight)
    compute.to(device)
    return compute


class SentencePlannerLoss(NMTLossCompute):

    def __init__(self, generator, symbols, vocab_size, label_smoothing=0.0, vocab=None, sentplan_loss_weight=0.0):
        super(SentencePlannerLoss, self).__init__(generator, symbols, vocab_size, label_smoothing, vocab)
        self.sentplan_loss_weight = sentplan_loss_weight

    def sharded_compute_loss(self, batch, output, shard_size, normalization):
        """ Adapted from LossComputeBase.sharded_compute_loss """
        batch_stats = Statistics()
        output, encoded_sents, generated_sents = output
        shard_state = self._make_shard_state(batch, output)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss = loss.div(float(normalization))
            sentplan_loss = mse_loss(encoded_sents, generated_sents)
            sentplan_loss_weighted = self.sentplan_loss_weight * sentplan_loss
            total_loss = loss + sentplan_loss_weighted
            total_loss.backward()

            # update statistics
            stats.xent_loss = loss.item()
            stats.sentplan_loss = sentplan_loss.item()
            stats.sentplan_loss_weighted = sentplan_loss_weighted.item()
            stats.total_loss = total_loss.item()
            batch_stats.update(stats)
        return batch_stats
