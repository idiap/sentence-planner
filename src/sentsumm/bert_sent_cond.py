#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Variant of the Bert decoder that conditions on a sentence representation. """

import torch
from torch import nn
from transformers.modeling_bert import BertLayerNorm, BertAttention, BertLayer, BertEncoder, BertModel


class BertSelfOutputSC(nn.Module):
    def __init__(self, config):
        super(BertSelfOutputSC, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense_sentence = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_sent_cond = config.output_sent_cond

    def forward(self, hidden_states, input_tensor, sentence_repr=None, alpha=None, baseline=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        outputs = ()

        if sentence_repr is None:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            # run sentence representation through linear + dropout, then add to hidden states and residual
            sentence_repr = self.dense_sentence(sentence_repr)
            sentence_repr = self.dropout(sentence_repr)

            # potentially scale tensors with alpha from baseline
            if alpha is not None and baseline is not None:
                baseline_hidden = baseline[0].unsqueeze(0)
                baseline_sent = baseline[1].unsqueeze(0)
                hidden_states = baseline_hidden + alpha * (hidden_states - baseline_hidden)
                sentence_repr = baseline_sent + alpha * (sentence_repr - baseline_sent)

            # potentially output (hidden states, input tensor, sentence representation)
            if self.output_sent_cond:
                outputs = (hidden_states, sentence_repr)

            # add 3 tensors and pass through layer norm
            hidden_states = self.LayerNorm(hidden_states + input_tensor + sentence_repr)
        return (hidden_states,) + outputs


class BertAttentionSC(BertAttention):
    def __init__(self, config):
        super(BertAttentionSC, self).__init__(config)
        self.output = BertSelfOutputSC(config)
        self.output_sent_cond = config.output_sent_cond

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, sentence_repr=None, alpha=None, baseline=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_outputs = self.output(self_outputs[0], hidden_states, sentence_repr, alpha, baseline)
        if self.output_sent_cond:
            outputs = (attention_outputs[0],) + attention_outputs[1:]  # add sent cond tensors if we output them
        else:
            outputs = (attention_outputs[0],) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLayerSC(BertLayer):
    def __init__(self, config):
        super(BertLayerSC, self).__init__(config)
        self.attention = BertAttentionSC(config)
        if self.is_decoder:
            self.crossattention = BertAttentionSC(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, sentence_repr=None, alpha=None, baseline=None):
        # in case cross-attention is used, we add the sentence representation there, otherwise we add it in the self attention
        if self.is_decoder and encoder_hidden_states is not None:
            sentence_repr_self_attn = None
        else:
            sentence_repr_self_attn = sentence_repr

        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, sentence_repr=sentence_repr_self_attn)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions or sent cond tensors if we output them

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask,
                sentence_repr, alpha, baseline
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions or sent cond tensors

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoderSC(BertEncoder):
    def __init__(self, config):
        num_hidden_layers = config.num_hidden_layers
        config.num_hidden_layers = 0  # don't initialize the layers in the superclass
        super(BertEncoderSC, self).__init__(config)
        config.num_hidden_layers = num_hidden_layers
        self.layer = nn.ModuleList([BertLayerSC(config) for _ in range(num_hidden_layers)])
        self.output_sent_cond = config.output_sent_cond

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, sentence_repr=None, alpha=None, baseline=None):
        all_sent_cond_outputs = ()
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            baseline_layer = baseline[i] if baseline is not None else None
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask, sentence_repr, alpha, baseline_layer)
            hidden_states = layer_outputs[0]

            if self.output_sent_cond:
                all_sent_cond_outputs = all_sent_cond_outputs + (layer_outputs[1:],)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_sent_cond:
            outputs = outputs + (all_sent_cond_outputs,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertModelSC(BertModel):
    def __init__(self, config):
        super(BertModelSC, self).__init__(config)
        self.encoder = BertEncoderSC(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                sentence_repr=None, alpha=None, baseline=None):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # AM: keep using uint8, same data type as `attention_mask`
                # causal_mask = causal_mask.to(torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder:
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(input_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask,
                                       sentence_repr=sentence_repr,
                                       alpha=alpha,
                                       baseline=baseline)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions), (hidden_tensors, embedding_output, sentence_repr)
