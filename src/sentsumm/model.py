#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" SentSumm model for sentence-level planning of summaries. """

import torch
import itertools
from torch import nn

from transformers import BertModel, BertConfig, BertForMaskedLM
from transformers.modeling_bert import BertEncoder

from .utils import parse_sentence_indices, find_sentence_boundaries
from .bert_sent_cond import BertModelSC
from .bert_output_crossattention import BertModelCA
from .bert_sent_head import BertModelSH


class TensorDict:
    """ Simplified dictionary with tensors as keys. """
    def __init__(self):
        self.store = dict()

    def _serialize_key(self, key):
        if not isinstance(key, torch.Tensor):
            raise ValueError('Unsupported key type: %s' % type(key))
        return repr(key.tolist())

    def put(self, key, value):
        self.store[self._serialize_key(key)] = value

    def get(self, key):
        """ Returns the stored value or None if the key is not present. """
        serialized_key = self._serialize_key(key)
        return self.store[serialized_key] if serialized_key in self.store else None


class WordEncoder(nn.Module):
    """ Encodes words/BPE to contextual embeddings with Bert. """
    def __init__(self, pretrained_dir, do_train=False, max_pos=512):
        super(WordEncoder, self).__init__()
        self.model = BertModel.from_pretrained(pretrained_dir)

        # handle positions > 512
        if (max_pos > 512):
            my_pos_embeddings = nn.Embedding(max_pos, self.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = \
                self.model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(max_pos - 512, 1)
            self.model.embeddings.position_embeddings = my_pos_embeddings

        # possibly set model to evaluation mode
        if not do_train:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x, segs, mask):
        top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        return top_vec


class SummaryEncoder(nn.Module):
    """
    Creates sentence representations from target input ids.
    Sentence representations may not look across sentence boundaries into the future (and possibly not into the past).
    """
    def __init__(self, word_encoder=None, pretrained_dir=None, do_train=False, max_pos=512, see_past=True):
        super(SummaryEncoder, self).__init__()
        self.see_past = see_past
        if word_encoder:
            self.encoder = word_encoder
        else:
            self.encoder = WordEncoder(pretrained_dir=pretrained_dir, do_train=do_train, max_pos=max_pos)

    def forward(self, input_ids, segment_ids, padding_mask, start_indices, end_indices, indices_mask):
        batch_size, input_size = input_ids.size()
        device = input_ids.device

        # create causal mask
        causal_mask = torch.zeros([batch_size, input_size, input_size], dtype=torch.uint8, device=device)
        if self.see_past:
            # tokens in a sentence may see tokens in previous sentences
            for i in range(batch_size):
                for start, end in zip(start_indices[i].tolist(), end_indices[i].tolist()):
                    if start == 0 and end == 0:
                        continue
                    causal_mask[i, start:end + 1, start:end + 1] = 1
        else:
            # tokens in each sentence can only see tokens in the same sentence
            for i in range(batch_size):
                for start, end in zip(start_indices[i].tolist(), end_indices[i].tolist()):
                    if start == 0 and end == 0:
                        continue
                    causal_mask[i, start:, :end + 1] = 1
        attention_mask = causal_mask * padding_mask[:, :, None]
        words_vec = self.encoder(x=input_ids, segs=segment_ids, mask=attention_mask)
        sents_vec = words_vec[torch.arange(words_vec.size(0)).unsqueeze(1), end_indices]
        sents_vec = sents_vec * indices_mask[:, :, None].float()
        return sents_vec


class SentenceGenerator(nn.Module):
    """
    Creates a summary sentence representation from the previous summary sentences' representations
    and the contextual Bert embeddings of the source document.
    """
    def __init__(self, config_args, pretrained_dir, see_past=True):
        super(SentenceGenerator, self).__init__()
        self.config = BertConfig.from_pretrained(pretrained_dir)
        self.config.is_decoder = True
        self.config.num_hidden_layers = config_args.sentgen_layers
        self.config.hidden_size = config_args.sentgen_hidden_size
        self.config.num_attention_heads = config_args.sentgen_heads
        self.config.intermediate_size = config_args.sentgen_ff_size
        self.config.hidden_dropout_prob = config_args.sentgen_hidden_dropout
        self.config.attention_probs_dropout_prob = config_args.sentgen_attn_dropout
        self.config.max_position_embeddings = config_args.max_pos
        self.model = BertEncoder(self.config)
        self.model.apply(self._init_weights)
        self.see_past = see_past

    _init_weights = BertModel._init_weights

    def forward(self, hidden_states, mask_hidden_states, encoder_hidden_states, mask_encoder_states):
        batch_size, seq_length, hidden_size = hidden_states.size()
        device = hidden_states.device

        # hidden states mask for padding tokens in hidden states and causal mask
        # [batch_size, num_heads, seq_length, seq_length]
        seq_ids = torch.arange(seq_length, device=device)
        if self.see_past:
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        else:
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) == seq_ids[None, :, None]
        attention_mask = causal_mask[:, None, :, :] * mask_hidden_states[:, None, None, :]
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0  # can now add the mask to scores

        # encoder states mask for padding tokens in encoder input
        # [batch_size, num_heads, seq_length, seq_length]
        encoder_attention_mask = mask_encoder_states[:, None, None, :]
        encoder_attention_mask = encoder_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0
        hidden_states, = self.model(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=[None] * self.config.num_hidden_layers,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return hidden_states


class BertLanguageModel(BertForMaskedLM):
    """ Language model with Bert, ignoring the MLM part and returning outputs instead of loss. """

    def __init__(self, config, from_pretrained=None, use_sent_cond=False, fix_sent_head=False):
        super(BertLanguageModel, self).__init__(config)
        if config.output_attentions:
            self.bert = BertModelCA(config)
        self.use_sent_cond = use_sent_cond
        if use_sent_cond:
            self.bert = BertModelSC(config)
            self.init_weights()
        if fix_sent_head:
            self.bert = BertModelSH(config)
            self.init_weights()
        if from_pretrained:
            cross_attentions = [l.crossattention for l in self.bert.encoder.layer]  # save initialized cross attentions
            self.bert = self.bert.from_pretrained(from_pretrained)  # loads an encoder Bert and deletes cross attentions
            self.bert.config.is_decoder = True  # restore the decoder setting and the cross attentions
            for l in self.bert.encoder.layer:
                l.is_decoder = True
            for ca, l in zip(cross_attentions, self.bert.encoder.layer):
                l.crossattention = ca

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                masked_lm_labels=None, encoder_hidden_states=None, encoder_attention_mask=None, lm_labels=None, sentence_repr=None,
                alpha=None, baseline=None):

        params = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
            'head_mask': head_mask,
            'inputs_embeds': inputs_embeds,
            'encoder_hidden_states': encoder_hidden_states,
            'encoder_attention_mask': encoder_attention_mask
        }
        if self.use_sent_cond:
            params['sentence_repr'] = sentence_repr
            params['alpha'] = alpha
            params['baseline'] = baseline
        outputs = self.bert(**params)

        if self.config.output_attentions or self.config.output_sent_cond:
            return outputs[0], outputs[2]
        else:
            return outputs[0]


class WordGenerator(nn.Module):
    """ Generates the words in the summary based on a summary sentence representation. """
    def __init__(self, config_args, pretrained_dir, load_pretrained=True, use_sent_cond=False, output_attentions=False, fix_sent_head=False):
        super(WordGenerator, self).__init__()
        config = BertConfig.from_pretrained(pretrained_dir)
        config.output_attentions = output_attentions
        config.output_sent_cond = hasattr(config_args, 'output_sent_cond') and config_args.output_sent_cond
        assert not (config.output_attentions and config.output_sent_cond)
        config.is_decoder = True
        if load_pretrained:
            self.model = BertLanguageModel(config, from_pretrained=pretrained_dir, use_sent_cond=use_sent_cond, fix_sent_head=fix_sent_head)
        else:
            config.hidden_size = config_args.dec_hidden_size
            config.num_hidden_layers = config_args.dec_layers
            config.num_attention_heads = config_args.dec_heads
            config.intermediate_size = config_args.dec_ff_size
            config.hidden_dropout_prob = config_args.dec_dropout
            config.attention_probs_dropout_prob = config_args.dec_dropout
            config.max_position_embeddings = config_args.max_pos
            self.model = BertLanguageModel(config, use_sent_cond=use_sent_cond, fix_sent_head=fix_sent_head)
        self.logits = self.model.cls

    def forward(self, hidden_states, mask_hidden_states, encoder_hidden_states, mask_encoder_states, sentence_repr=None,
                alpha=None, baseline=None):
        return self.model(
            input_ids=hidden_states,
            attention_mask=mask_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=mask_encoder_states,
            sentence_repr=sentence_repr,
            alpha=alpha,
            baseline=baseline,
        ), None  # return None for scores to match API

class SentSumm(nn.Module):
    """ SentSumm model. """
    def __init__(self, args, device, symbols, checkpoint=None, bert_from_extractive=None):
        super(SentSumm, self).__init__()
        max_pos = 512
        word_enc = WordEncoder(pretrained_dir=args.pretrained_dir, do_train=args.finetune_enc, max_pos=max_pos)
        if bert_from_extractive:
            word_enc.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)
        self.enc_article = word_enc
        if args.share_word_enc:
            self.enc_summary = SummaryEncoder(word_encoder=word_enc, see_past=args.see_past_words)
        else:
            self.enc_summary = SummaryEncoder(pretrained_dir=args.pretrained_dir, do_train=args.finetune_enc, max_pos=max_pos, see_past=args.see_past_words)

        # check dimensions are correct
        encoder_hidden_size = self.enc_article.model.config.hidden_size
        assert encoder_hidden_size == self.enc_summary.encoder.model.config.hidden_size, "Article and summary encoders have different hidden sizes"

        # project encoder output if it has different dimensionality from decoder (sentgen or wordgen)
        self.enc2sentgen = None
        self.enc2wordgen = None
        if encoder_hidden_size != args.sentgen_hidden_size:
            self.enc2sentgen = nn.Linear(encoder_hidden_size, args.sentgen_hidden_size)
        if args.dec_hidden_size == args.sentgen_hidden_size:
            self.enc2wordgen = self.enc2sentgen
        elif encoder_hidden_size != args.dec_hidden_size:
            self.enc2wordgen = nn.Linear(encoder_hidden_size, args.dec_hidden_size)

        # project sentence planner output if it has different dimensionality from decoder
        self.sentgen2wordgen = None
        if args.sentgen_hidden_size != args.dec_hidden_size:
            self.sentgen2wordgen = nn.Linear(args.sentgen_hidden_size, args.dec_hidden_size)

        if not args.use_lm_only and not args.no_sent_repr:
            self.gen_summary_sents = SentenceGenerator(args, args.pretrained_dir, see_past=args.see_past_sents)
        self.gen_summary_words = WordGenerator(args, args.pretrained_dir, load_pretrained=args.pretrain_dec, use_sent_cond=args.use_sent_cond, output_attentions=args.output_attentions, fix_sent_head=args.fix_sent_head)
        self.generator = nn.Sequential(self.gen_summary_words.logits, nn.LogSoftmax(dim=-1))
        self.first_sent_emb = torch.zeros([1, args.sentgen_hidden_size], device=device)
        if args.train_first_sent_emb:
            self.first_sent_emb = nn.Parameter(self.first_sent_emb)
        self.use_dec_src_attn = args.use_dec_src_attn
        self.pretrain_dec = args.pretrain_dec
        self.use_sent_cond = args.use_sent_cond
        self.no_sent_repr = args.no_sent_repr
        self.use_lm_only = args.use_lm_only
        self.symbols = symbols
        self.vocab_size = word_enc.model.config.vocab_size
        self.sentplan_loss_weight = args.sentplan_loss_weight
        self.output_sent_cond = hasattr(args, 'output_sent_cond') and args.output_sent_cond

        # fix_sent_head implies use_dec_src_attn
        assert args.use_dec_src_attn or not args.fix_sent_head

        # potentially load checkpoint
        if checkpoint:
            self.load_state_dict(checkpoint['model'], strict=not args.load_partial)

        # potentially tie embeddings
        if args.tie_input_embs:
            # word encoder with word decoder input embeddings
            self.gen_summary_words.model.get_input_embeddings().weight = word_enc.model.get_input_embeddings().weight
        if args.tie_decoder_embs:
            # word decoder input with output embeddings
            self.gen_summary_words.model.get_output_embeddings().weight = self.gen_summary_words.model.get_input_embeddings().weight

        # push model to GPU if applicable
        self.to(device)
        self.device = device

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls, alpha=None, baseline=None):
        # sanity check: language model only (is the attention mask not causal in the decoder?)
        if self.use_lm_only:
            return self.gen_summary_words.forward(
                hidden_states=tgt[:, :-1],
                mask_hidden_states=mask_tgt[:, 1:],
                encoder_hidden_states=None,
                mask_encoder_states=None,
            )

        encoded_src = self.enc_article(src, segs=segs, mask=mask_src)

        # potentially project encoded article
        encoded_src_sentgen = encoded_src_wordgen = encoded_src
        if self.enc2sentgen is not None:
            encoded_src_sentgen = self.enc2sentgen(encoded_src)
        if self.enc2wordgen is not None:
            encoded_src_wordgen = self.enc2wordgen(encoded_src)

        # baseline comparison: use only attention over encoded article and ignore sentence representation
        if self.no_sent_repr:
            return self.gen_summary_words.forward(
                hidden_states=tgt[:, :-1],
                mask_hidden_states=mask_tgt[:, 1:],
                encoder_hidden_states=encoded_src_wordgen,
                mask_encoder_states=mask_src,
            )

        # encode summary sentences
        encoded_sents, tgt_starts, tgt_ends, indices_mask = self._encode_sentence(tgt, mask_tgt)
        if self.enc2sentgen is not None:
            encoded_sents = self.enc2sentgen(encoded_sents)

        # shift encoded sentences by one
        batch_size, num_sents, hidden_size = encoded_sents.size()
        first_hidden_state = self.first_sent_emb.repeat(batch_size, 1, 1)
        shifted_sents = torch.cat([first_hidden_state, encoded_sents[:, :-1, :]], dim=1)

        # generate summary sentences
        generated_sents = self.gen_summary_sents(
            shifted_sents,
            mask_hidden_states=indices_mask,
            encoder_hidden_states=encoded_src_sentgen,
            mask_encoder_states=mask_src
        )
        if self.sentgen2wordgen is not None:
            generated_sents = self.sentgen2wordgen(generated_sents)

        # second forward path for sentplan loss (to be able to detach encoded_src and encoded_sents)
        if self.sentplan_loss_weight and self.training:
            generated_sents_sentplan = self.gen_summary_sents(
                shifted_sents.detach(),
                mask_hidden_states=indices_mask,
                encoder_hidden_states=encoded_src_sentgen.detach(),
                mask_encoder_states=mask_src
            )

        # prepare word generator's hidden states (with teacher forcing, starts with CLS as start token)
        hidden_states = tgt[:, :-1]
        mask_hidden_states = mask_tgt[:, 1:]

        # mask: attend only to current generated sentences
        # [batch_size, tgt_length, num_sents]
        tgt_lengths = torch.sum(mask_hidden_states, dim=1, keepdim=True)
        mask_tensor = torch.eye(generated_sents.size(1), dtype=torch.uint8, device=self.device)
        padding_tensor = torch.zeros([1, generated_sents.size(1)], dtype=torch.uint8, device=self.device)
        mask_tensor = torch.cat([mask_tensor, padding_tensor], dim=0)  # add padding
        repeats = (tgt_ends - tgt_starts + 1) * indices_mask.to(dtype=torch.long)
        repeats = torch.cat([repeats, hidden_states.size(1) - tgt_lengths], dim=1)  # add repeats for padding

        # reduce last non-padding repeat by one, since we don't want to predict a next token for the EOQ hidden state
        for r in repeats:
            i = -2  # r[-1] is padding repetitions, start at -2
            while r[i] == 0:
                i -= 1
            r[i] -= 1

        # either repeat sentence representation to add in decoder, or mask for decoder cross-attention
        if self.use_sent_cond:
            assert generated_sents.dim() == 3  # [batch_size, num_sents, hidden_size]
            padding_repr = torch.zeros([generated_sents.size(0), 1, generated_sents.size(2)], device=self.device)
            sentence_repr = torch.cat([generated_sents, padding_repr], dim=1)
            sentence_repr = torch.stack([sentence_repr[i].repeat_interleave(repeats[i], dim=0) for i in range(batch_size)])

            # potentially add encoder hidden states to use in decoder cross-attention
            encoder_hidden_states = None
            mask_encoder_states = None
            if self.use_dec_src_attn:
                encoder_hidden_states = encoded_src_wordgen
                mask_encoder_states = mask_src[:, None, :] * mask_tgt[:, 1:, None]

            outputs = self.gen_summary_words.forward(
                hidden_states=hidden_states,
                mask_hidden_states=mask_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                mask_encoder_states=mask_encoder_states,
                sentence_repr=sentence_repr,
                alpha=alpha,
                baseline=baseline,
            )
        else:
            mask_gen_sents = torch.stack([mask_tensor.repeat_interleave(repeats[i], dim=0) for i in range(batch_size)])
            mask_gen_sents = mask_gen_sents * indices_mask[:, None, :]  # mask padding tokens in generated sentences

            # possibly concatenate encoded source article
            encoder_hidden_states = generated_sents
            if self.use_dec_src_attn:
                encoder_hidden_states = torch.cat([generated_sents, encoded_src_wordgen], dim=1)
                mask_src_tgt = mask_src[:, None, :] * mask_tgt[:, 1:, None]
                mask_gen_sents = torch.cat([mask_gen_sents, mask_src_tgt], dim=2)

            # generate summary words
            outputs = self.gen_summary_words.forward(
                hidden_states=hidden_states,
                mask_hidden_states=mask_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                mask_encoder_states=mask_gen_sents,
            )
        if self.sentplan_loss_weight and self.training:
            return (outputs[0], encoded_sents.detach(), generated_sents_sentplan), None
        else:
            return outputs

    def get_pretrained_params(self):
        """ Returns the named parameters that have been initialized from a checkpoint. """
        named_params = itertools.chain(
            self.enc_article.named_parameters(prefix='enc_article'),
            self.enc_summary.named_parameters(prefix='enc_summary'),
            self.gen_summary_words.named_parameters(prefix='gen_summary_words') if self.pretrain_dec else [],
        )
        return named_params

    def get_newly_trained_params(self):
        """ Returns the named parameters that have been randomly initialized. """
        named_params = itertools.chain(
            self.gen_summary_sents.named_parameters(prefix='gen_summary_sents') if hasattr(self, 'gen_summary_sents') else [],
            self.gen_summary_words.named_parameters(prefix='gen_summary_words') if not self.pretrain_dec else [],
            [('first_sent_emb', self.first_sent_emb)],
        )
        return named_params

    def _encode_sentence(self, tgt, mask_tgt):
        """ Encodes the sentences in `tgt`. """
        # find EOS and EOQ separators
        tgt_starts, tgt_ends, indices_mask = find_sentence_boundaries(self.symbols, tgt)

        # convert to tensors
        tgt_starts = torch.tensor(tgt_starts, dtype=torch.long, device=self.device)
        tgt_ends = torch.tensor(tgt_ends, dtype=torch.long, device=self.device)
        indices_mask = torch.tensor(indices_mask, dtype=torch.uint8, device=self.device)

        # encode sentences
        encoded_sents = self.enc_summary(
            input_ids=tgt,
            segment_ids=None,
            padding_mask=mask_tgt,
            start_indices=tgt_starts,
            end_indices=tgt_ends,
            indices_mask=indices_mask,
        )
        return encoded_sents, tgt_starts, tgt_ends, indices_mask

    ### Methods for inference ###
    def _find_indices(self, sequence, symbol):
        """ Finds the indices of `symbol` in tensor `sequence`. """
        assert sequence.dim() == 1 or sequence.dim() == 2 and sequence.size(0) == 1
        if sequence.dim() == 2:
            sequence = sequence.squeeze(0)
        return (sequence == symbol).nonzero()[:, 0]

    def _generate_next_sentence(self, tgt, mask_tgt, encoded_src, mask_encoded_src):
        """ Generates the next sentence representation based on the previous sentences. """
        assert encoded_src.dim() == 3
        batch_size = encoded_src.size(0)
        first_sents = self.first_sent_emb.repeat(batch_size, 1, 1)
        if tgt is not None:
            assert tgt.dim() == 2
            encoded_sents, _, _, _ = self._encode_sentence(tgt, mask_tgt)
            if self.enc2sentgen is not None:
                encoded_sents = self.enc2sentgen(encoded_sents)
            input_sents = torch.cat([first_sents, encoded_sents], dim=1)
        else:
            # for first sentence representation, input_sents is just the first sentence embedding
            input_sents = first_sents
        mask_input_sents = torch.ones(input_sents.size()[:2], dtype=torch.uint8, device=self.device)
        output_sents = self.gen_summary_sents(
            hidden_states=input_sents,
            mask_hidden_states=mask_input_sents,
            encoder_hidden_states=encoded_src,
            mask_encoder_states=mask_encoded_src,
        )
        if self.sentgen2wordgen is not None:
            output_sents = self.sentgen2wordgen(output_sents)
        return output_sents[:, -1]

    def _get_next_sentence(self, tgt, encoded_src, mask_src):
        """ Get the next sentence representation either from the cache or by generating it. """
        if self.symbols['EOS'] not in tgt:
            # no sentence has finished yet; use first sentence representation
            return self.generated_sents.get(torch.tensor([]))

        # there are finished previous sentences; try to retrieve their generated representation
        last_sep = self._find_indices(tgt, self.symbols['EOS']).tolist()[-1]
        completed_sents = tgt[:last_sep + 1]
        cached_sent_repr = self.generated_sents.get(completed_sents)
        if cached_sent_repr is not None:
            return cached_sent_repr

        # we have no cached sentence representation for these previous sentences
        tgt_sents = completed_sents.unsqueeze(0)
        mask_tgt_sents = torch.ones_like(tgt_sents, dtype=torch.uint8)
        sent_repr = self._generate_next_sentence(tgt_sents, mask_tgt_sents, encoded_src, mask_src)
        self.generated_sents.put(completed_sents, sent_repr)
        return sent_repr

    def _generate_next_word(self, tgt, mask_tgt, sentence_repr, encoded_src, mask_src):
        """ Generates the next word based on the current sentence representation and the previous words."""
        assert tgt.dim() == 2
        batch_size, tgt_len = tgt.size()
        if self.no_sent_repr:
            encoder_hidden_states = encoded_src
            mask_encoder_states = mask_src
        elif self.use_sent_cond:
            encoder_hidden_states = None
            mask_encoder_states = None
            if self.use_dec_src_attn:
                encoder_hidden_states = encoded_src.expand(batch_size, -1, -1)
                mask_encoder_states = mask_src[:, None, :] * mask_tgt[:, :, None]
        else:
            encoder_hidden_states = sentence_repr
            mask_encoder_states = torch.ones([batch_size, tgt_len, 1], dtype=torch.uint8, device=self.device) * mask_tgt[:, :, None]
            if self.use_dec_src_attn:
                encoder_hidden_states = torch.cat([encoder_hidden_states, encoded_src.expand(batch_size, -1, -1)], dim=1)
                mask_src_tgt = mask_src[:, None, :] * mask_tgt[:, :, None]
                mask_encoder_states = torch.cat([mask_encoder_states, mask_src_tgt], dim=2)

        params = {
            'hidden_states': tgt,
            'mask_hidden_states': mask_tgt,
            'encoder_hidden_states': encoder_hidden_states,
            'mask_encoder_states': mask_encoder_states,
        }
        if self.use_sent_cond:
            params['sentence_repr'] = sentence_repr
        return self.gen_summary_words.forward(**params)

    def inference_init(self, src, segs, mask_src):
        """ Initialize inference. """
        # reset the encoded and generated sentence representations
        self.generated_sents = TensorDict()

        # encode the article words and set up the output words with the start token
        encoded_src = self.enc_article(src, segs=segs, mask=mask_src)
        encoded_src_sentgen = encoded_src
        if self.enc2sentgen is not None:
            encoded_src_sentgen = self.enc2sentgen(encoded_src)

        # create the first sentence representation
        if not self.no_sent_repr:
            first_sent_repr = self._generate_next_sentence(None, None, encoded_src_sentgen, mask_src)
            self.generated_sents.put(torch.tensor([]), first_sent_repr)
        return encoded_src

    def inference_step(self, encoded_src, mask_src, tgt, mask_tgt):
        """
        Run one step of inference. Currently only works for batch size of 1.
        encoded_src: [batch_size x seq_len x hidden_size]
        tgt: [(batch_size * beam_size) x seq_len_so_far]
        """
        assert encoded_src.dim() == 3 and encoded_src.size(0) == 1

        # potentially project encoder output dimensions
        encoded_src_sentgen = encoded_src_wordgen = encoded_src
        if self.enc2sentgen is not None:
            encoded_src_sentgen = self.enc2sentgen(encoded_src)
        if self.enc2wordgen is not None:
            encoded_src_wordgen = self.enc2wordgen(encoded_src)

        # get sentence representations for previous sentences (from cache or generate)
        generated_sents = None
        if not self.no_sent_repr:
            generated_sents = [self._get_next_sentence(t, encoded_src_sentgen, mask_src) for t in tgt]
            generated_sents = torch.stack(generated_sents, dim=0)

        # generate the next word
        generated_words = self._generate_next_word(tgt, mask_tgt, generated_sents, encoded_src_wordgen, mask_src)
        return generated_words
