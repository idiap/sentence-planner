#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Bert decoder variant with one head focused on the sentence representation. """

import math
import torch
from torch import nn
from transformers.modeling_bert import BertSelfAttention, BertAttention, BertLayer, BertEncoder, BertModel


def find_num_sentences(mask_tensor):
    """ Determines the number of sentences based on the first entry of the encoder attention mask. """
    # the mask tensor is a concatenation of the mask for sentence representations and article BPE token embeddings
    # it should look like: [0, -1e9, -1e9, -1e9, 0, 0, ...] for an example with 4 sentences
    assert mask_tensor[0] == 0
    i = 1
    while mask_tensor[i] != 0:
        i += 1
    assert i < 50  # something went wrong, we never have that many sentences
    return i


class BertSelfAttentionSH(BertSelfAttention):

    def large_negative_value(self):
        dtype = next(self.parameters()).dtype
        if dtype == torch.float16:
            return -1e4
        elif dtype == torch.float32:
            return -1e9
        else:
            raise ValueError('')

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        # make sure we are in decoder
        assert encoder_hidden_states is not None
        assert encoder_attention_mask is not None

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Changed: Mask article for first head, mask sentence representations for other heads
            batch_size, num_heads, _, _ = attention_scores.size()
            attention_mask = attention_mask.expand(-1, num_heads, -1, -1).clone()  # clone to allocate memory
            num_sentences = find_num_sentences(attention_mask[0, 0, 0])
            attention_mask[:, 0, :, num_sentences:] = self.large_negative_value()
            attention_mask[:, 1:, :, :num_sentences] = self.large_negative_value()
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertAttentionSH(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = BertSelfAttentionSH(config)


class BertLayerSH(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        assert self.is_decoder
        self.crossattention = BertAttentionSH(config)


class BertEncoderSH(BertEncoder):
    def __init__(self, config):
        num_hidden_layers = config.num_hidden_layers
        config.num_hidden_layers = 0  # don't initialize the layers in the superclass
        super().__init__(config)
        config.num_hidden_layers = num_hidden_layers
        self.layer = nn.ModuleList([BertLayerSH(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # changed: output cross-attentions

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertModelSH(BertModel):
    def __init__(self, config):
        num_hidden_layers = config.num_hidden_layers
        config.num_hidden_layers = 0  # don't initialize the layers in the superclass
        super().__init__(config)
        config.num_hidden_layers = num_hidden_layers
        self.encoder = BertEncoderSH(config)
        self.init_weights()
