# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BART model, ported from the fairseq repo."""
import logging
import math
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from .activations import ACT2FN
from .configuration_bart import BartConfig
from .file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
)
from .modeling_utils import PreTrainedModel
from .encoder import TransformerInterEncoder
from .new_layer import TransformerDecoder
from .generator import Beam

logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "BartTokenizer"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/mbart-large-en-ro",
    # See all BART models at https://huggingface.co/models?filter=bart
]


BART_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matters related to general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.

"""
BART_GENERATION_EXAMPLE = r"""
    Summarization example::

        from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

        # see ``examples/summarization/bart/run_eval.py`` for a longer example
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
               Indices of input sequence tokens in the vocabulary. Use BartTokenizer.encode to produce them.
            Padding will be ignored by default should you provide it.
            Indices can be obtained using :class:`transformers.BartTokenizer.encode(text)`.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
            `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention of the decoder.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for translation and summarization training. By default, the model will create this tensor by shifting the input_ids right, following the paper.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
            Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
            If you want to change padding behavior, you should read :func:`~transformers.modeling_bart._prepare_decoder_inputs` and modify.
            See diagram 1 in the paper for more info on the default strategy
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
"""


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


def _prepare_bart_decoder_inputs(
    config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask


class PretrainedBartModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


# Helper Modules


class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = SelfAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, output_attentions=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask, output_attentions=output_attentions
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn_weights


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()
        # self.sen_layer_num = sen_layer_num

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

        self.hidden_size = config.hidden_size

    def forward(self,
                pair_input_ids,
                pair_attention_mask,
                sentence_input_id,
                sentence_attention_mask,
                sentence_length,
                pairs_list,
                passage_length,
                pairs_num,
                para_input_id,
                para_attention_mask,
                max_sentence_length,
                cuda,
                output_attentions=False,
                output_hidden_states=False):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        if para_attention_mask is not None:
            para_attention_mask = invert_mask(para_attention_mask)

        ######### for paragraphs #########
        inputs_embeds = self.embed_tokens(para_input_id) * self.embed_scale
        embed_pos = self.embed_positions(para_input_id)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            # if output_hidden_states:
            #     encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, para_attention_mask, output_attentions=output_attentions)


        if self.layer_norm:
            x = self.layer_norm(x)
        # if output_hidden_states:
        #     encoder_states.append(x)


        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        return x, encoder_states, all_attentions


class BartEncoder_up2(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, up_layer_num, embed_tokens):
        super().__init__()
        self.up_layer_num = up_layer_num

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(up_layer_num)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

        self.hidden_size = config.hidden_size

    def para_transform_pair(self, para_emb, sentence_length, pair_position_emb, pairs_list,
                           passage_length, pairs_num, max_sent_length, cuda=None):

        batch, max_pair_num, _ = pairs_list.size()
        batch_max_pair_num, max_pair_length, _ = pair_position_emb.size()
        batch, max_sen_num = sentence_length.size()

        ########## get single sentence feature
        sent_hidden_embed = torch.zeros(batch, max_sen_num, max_sent_length, self.hidden_size).to(int(cuda[-1]))
        for batch_id in range(batch):
            sen_num = passage_length[batch_id]
            # para_emb_batchi = para_emb[batch_id]
            true_sen_len = sentence_length[batch_id][:sen_num]

            sent_len = 1
            for sent_i in range(sen_num):
                sent_len_i = true_sen_len[sent_i]-1
                sent_hidden_embed[batch_id][sent_i][:sent_len_i] = para_emb[batch_id][sent_len:sent_len+sent_len_i, :]
                sent_len = sent_len + sent_len_i

        sen_emb_new = sent_hidden_embed

        ### cls_output_matrix ###
        pair_hidden_embed = torch.zeros(batch, max_pair_num, max_pair_length, self.hidden_size).to(int(cuda[-1]))

        for batch_id in range(batch):
            current_true_pair_num = pairs_num[batch_id]
            true_pair_list = pairs_list[batch_id][:current_true_pair_num]

            sen_num = passage_length[batch_id]
            true_sen_emb = sen_emb_new[batch_id][:sen_num]

            true_sen_len = sentence_length[batch_id][:sen_num]

            for index, pair_id in enumerate(true_pair_list):
                pair_id_0 = pair_id[0]
                pair_id_1 = pair_id[1]

                sen_len_0 = true_sen_len[pair_id_0]
                sen_len_1 = true_sen_len[pair_id_1]

                sen_0_true_embed = true_sen_emb[pair_id_0][:sen_len_0-1, :]
                sen_1_true_embed = true_sen_emb[pair_id_1][:sen_len_1-1, :]

                cls_new_embed = torch.mean(torch.cat([sen_0_true_embed, sen_1_true_embed], 0), 0)

                pair_hidden_embed[batch_id][index][0] = cls_new_embed
                pair_hidden_embed[batch_id][index][1:sen_len_0] = sen_0_true_embed[:]
                pair_hidden_embed[batch_id][index][sen_len_0:sen_len_0 + sen_len_1 - 1] = sen_1_true_embed[:]

        pair_hidden_embed = pair_hidden_embed.reshape(batch_max_pair_num, max_pair_length, self.hidden_size)

        final_pair_embed = pair_hidden_embed + pair_position_emb

        return final_pair_embed


    def forward(self,
                pair_input_ids,
                pair_attention_mask,
                sentence_input_id,
                sentence_attention_mask,
                sentence_length,
                pairs_list,
                passage_length,
                pairs_num,
                para_input_id,
                para_attention_mask,
                max_sentence_length,
                top_rep,
                cuda,
                output_attentions=False,
                output_hidden_states=False):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        x = top_rep


        ######### for pairs #########
        bsz_parnum, _ = pair_input_ids.size()
        pair_embed_pos = self.embed_positions(pair_input_ids).unsqueeze(0).repeat(bsz_parnum, 1, 1)

        x = x.transpose(0, 1)
        encoder_states, all_attentions = [], []
        x = x.transpose(0, 1)
        x = self.para_transform_pair(x, sentence_length, pair_embed_pos, pairs_list,
                                     passage_length, pairs_num, max_sentence_length, cuda)
        x = x.transpose(0, 1)
        encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        return x, encoder_states, all_attentions

class DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_cached_states=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """
        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        return x, next_cache, all_hidden_states, list(all_self_attns)


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache


class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute key and value if they are static
                if static_kv:
                    key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training,)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        key_padding_mask = self._cat_prev_key_padding_mask(
            key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv
        )
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)

        elif key_padding_mask is not None:
            filler = torch.zeros(
                batch_size,
                src_len - key_padding_mask.size(1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead

    def __init__(
        self, input_dim, inner_dim, num_classes, pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None
        num_embeddings += offset
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _filter_out_falsey_values(tup) -> Tuple:
    """Remove entries that are None or [] from an iterable."""
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)


# Public API
def _get_shape(t):
    return getattr(t, "shape", None)


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.", BART_START_DOCSTRING,
)
class BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config,  self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="facebook/bart-large")
    def forward(
            self,
            pair_input_ids,
            pair_attention_mask,
            sentence_input_id,
            sentence_attention_mask,
            sentence_length,
            pairs_list,
            passage_length,
            pairs_num,
            para_input_id,
            para_attention_mask,
            max_sentence_length,
            cuda=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            encoder_outputs=None,
    ):

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                pair_input_ids,
                pair_attention_mask,
                sentence_input_id,
                sentence_attention_mask,
                sentence_length,
                pairs_list,
                passage_length,
                pairs_num,
                para_input_id,
                para_attention_mask,
                max_sentence_length,
                cuda,
                output_attentions=True,
                output_hidden_states=True,
            )

        x, all_encoder_states, all_attentions = encoder_outputs
        pooled_output = x[:, 0]
        outputs = x, pooled_output, all_encoder_states, all_attentions

        return outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.", BART_START_DOCSTRING,
)
class BartModel_up2(PretrainedBartModel):
    def __init__(self, config: BartConfig, up_layer_num):
        super().__init__(config, up_layer_num)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder_up2(config, up_layer_num, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="facebook/bart-large")
    def forward(
            self,
            pair_input_ids,
            pair_attention_mask,
            sentence_input_id,
            sentence_attention_mask,
            sentence_length,
            pairs_list,
            passage_length,
            pairs_num,
            para_input_id,
            para_attention_mask,
            max_sentence_length,
            top_vec,
            cuda=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            encoder_outputs=None,
    ):

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                pair_input_ids,
                pair_attention_mask,
                sentence_input_id,
                sentence_attention_mask,
                sentence_length,
                pairs_list,
                passage_length,
                pairs_num,
                para_input_id,
                para_attention_mask,
                max_sentence_length,
                top_vec,
                cuda,
                output_attentions=True,
                output_hidden_states=True,
            )

        x, all_encoder_states, all_attentions = encoder_outputs
        pooled_output = x[:, 0]
        outputs = x, pooled_output, all_encoder_states, all_attentions

        return outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
            The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out[:, 0 : dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))  # This line breaks for odd n_pos
        out[:, dim // 2 :] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)


class HierarchicalAttention(nn.Module):
    def __init__(self, config,):
        super(HierarchicalAttention, self).__init__()

        self.linear_in_2 = nn.Linear(config.d_model, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.sentence_tran = nn.Linear(config.d_model, config.d_model)
        self.sentence_tran_2 = nn.Linear(config.d_model, 1)

        self.dropout = nn.Dropout(config.dropout)

        ### pairwise loss ###
        self.pairwise_relationship = nn.Linear(config.d_model, 2)
        self.h1_relationship = nn.Linear(config.d_model, 2)
        self.h2_relationship = nn.Linear(config.d_model, 2)


    def forward(self, sequence_output, pairs_list, passage_length, pairs_num,
                sep_positions, cuda, mask_cls, cls_pooled_output):
        expanded_sample_num, max_len, hidden_size = sequence_output.size()

        batch, max_pair_num, _ = sep_positions.size()
        _, max_num_sen = mask_cls.size()

        sep_positions = sep_positions.reshape(expanded_sample_num, 2)

        top_vec = sequence_output
        top_vec_tran = self.sentence_tran(top_vec)
        top_vec_tran_tan = torch.tanh(top_vec_tran)
        top_vec_tran_score = self.sentence_tran_2(top_vec_tran_tan).squeeze(-1)
        selected_masks_all = []

        for sep in sep_positions:

            sep_0 = int(sep[0].cpu())
            sep_1 = int(sep[1].cpu())

            select_mask_0 = [0] + [1]*sep_0 + [0]*(max_len-sep_0-1)
            select_mask_1 = [0]*(sep_0+1) + [1]*(sep_1-sep_0) + [0]*(max_len-sep_1-1)

            new_select_mask = [select_mask_0, select_mask_1]
            selected_masks_all.append(new_select_mask)

        selected_masks_all = torch.FloatTensor(selected_masks_all).to(int(cuda[-1]))

        attention_mask = selected_masks_all.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0

        selected_masks_all_t = selected_masks_all.permute(0,2,1)
        select_score_t = selected_masks_all_t*top_vec_tran_score[:, :, None]
        select_score = select_score_t.permute(0,2,1)
        attention_scores = select_score + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        mix = torch.bmm(attention_probs, top_vec)
        mix = mix.reshape(batch, max_pair_num, 2, hidden_size)
        final_seq_matriax = torch.zeros(batch, max_num_sen, hidden_size).to(int(cuda[-1]))

        ### relative score ####
        cls_score = self.pairwise_relationship(cls_pooled_output)
        cls_score_matrix_nn = torch.zeros(batch, max_num_sen, max_num_sen, 2).to(int(cuda[-1]))
        cls_score_batch = cls_score.reshape(batch, max_pair_num, 2)

        ### history score ###
        cls_score_his1 = self.h1_relationship(cls_pooled_output)
        cls_score_matrix_nn_his1 = torch.zeros(batch, max_num_sen, max_num_sen, 2).to(int(cuda[-1]))
        cls_score_batch_his1 = cls_score_his1.reshape(batch, max_pair_num, 2)

        cls_score_his2 = self.h2_relationship(cls_pooled_output)
        cls_score_matrix_nn_his2 = torch.zeros(batch, max_num_sen, max_num_sen, 2).to(int(cuda[-1]))
        cls_score_batch_his2 = cls_score_his2.reshape(batch, max_pair_num, 2)

        ### cls_output_matrix ###
        cls_output_matrix_nn = torch.zeros(batch, max_num_sen, max_num_sen, hidden_size).to(int(cuda[-1]))
        cls_pooled_output_batch = cls_pooled_output.reshape(batch, max_pair_num, hidden_size)

        for idx_i, length in enumerate(passage_length):
            current_true_pair_num = pairs_num[idx_i]
            true_pair_list = pairs_list[idx_i][:current_true_pair_num]

            count = [0 for item in range(length)]
            edge_num = int(len(true_pair_list)/length*2)

            sentence_tensor_i = torch.zeros(length, edge_num, hidden_size).to(int(cuda[-1]))

            for idx_j, pair in enumerate(true_pair_list):

                current_place_0 = count[pair[0]]
                sentence_tensor_i[pair[0]][current_place_0] = mix[idx_i][idx_j][0]
                count[pair[0]] = count[pair[0]] + 1

                current_place_1 = count[pair[1]]
                sentence_tensor_i[pair[1]][current_place_1]=mix[idx_i][idx_j][1]
                count[pair[1]] = count[pair[1]] + 1

                ### cls output ###
                cls_output_matrix_nn[idx_i][pair[0]][pair[1]] = cls_pooled_output_batch[idx_i][idx_j]

                ### relative score ###
                cls_score_matrix_nn[idx_i][pair[0]][pair[1]] = cls_score_batch[idx_i][idx_j]

                ### history score ###
                cls_score_matrix_nn_his1[idx_i][pair[0]][pair[1]] = cls_score_batch_his1[idx_i][idx_j]
                cls_score_matrix_nn_his2[idx_i][pair[0]][pair[1]] = cls_score_batch_his2[idx_i][idx_j]

            sample = sentence_tensor_i
            passage_num, edge_num, hidden_size = sample.size()


            query_2 = sample.view(passage_num * edge_num, hidden_size).to(int(cuda[-1]))
            query_2 = self.linear_in_2(query_2)
            query_2 = query_2.reshape(passage_num, edge_num)


            attention_scores = query_2.unsqueeze(dim=1)

            attention_scores = attention_scores.view(passage_num * 1, edge_num)
            attention_weights = self.softmax(attention_scores)
            attention_weights = attention_weights.view(passage_num, 1, edge_num)
            mix_new = torch.bmm(attention_weights, sample).squeeze()

            final_seq_matriax[idx_i][:length] = mix_new

        return mix, final_seq_matriax, cls_output_matrix_nn, cls_score, cls_score_matrix_nn, \
               cls_score_matrix_nn_his1, cls_score_matrix_nn_his2


@add_start_docstrings(
    """Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks. """,
    BART_START_DOCSTRING,
)
class BartForOrdering_student(PretrainedBartModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        config.encoder_layers = 12
        up_layer_num = 1
        self.model = BartModel(config)

        self.model_up2 = BartModel_up2(config, up_layer_num)

        self.classification_head = BartClassificationHead(
            config.d_model, config.d_model, config.num_labels, config.classif_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.d_model
        self.key_linear = nn.Linear(config.d_model * 2, config.d_model)
        self.query_linear = nn.Linear(config.d_model, config.d_model)
        self.tanh_linear = nn.Linear(config.d_model, 1)


        self.encoder = TransformerInterEncoder(config.d_model, 2048, 8, 0.1, 2)
        self.critic = None

        ### pairwise ###
        self.two_level_encoder = HierarchicalAttention(config)

        # ### pairwise loss ###
        self.pairwise_loss_lam = 0.6

        ### pairwise decoder ###
        d_pair_posi = config.d_model
        self.pw_k = nn.Linear(d_pair_posi * 2, config.d_model, False)

        #### transformer decoder ####
        self.tran_decoder = TransformerDecoder(position_embed_hidden = config.d_model, n_position = 30, 
            d_model = config.d_model, d_inner = 3072,
            n_layers = 1, n_head = 8, dropout = 0.1)

        self.fut_tran_forw = nn.Linear(config.d_model, 1)
        self.fut_tran_back = nn.Linear(config.d_model, 1)

    def rela_encode(self, cls_output_matrix_nn, cls_score_matrix_nn):
        return cls_output_matrix_nn

    def history_encode(self, cls_output_matrix_nn, cls_score_matrix_nn_his1, cls_score_matrix_nn_his2):
        return cls_output_matrix_nn, cls_output_matrix_nn

    def equip(self, critic):
        self.critic = critic

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                pairs_list=None, passage_length=None, pairs_num=None, sep_positions=None, ground_truth=None, 
                mask_cls=None, pairwise_labels=None,
                sentence_input_id=None, sentence_attention_mask=None, sentence_length=None,
                para_input_id=None, para_attention_mask=None, max_sentence_length=None,
                cuda=None, head_mask=None):

        document_matrix, enc_output, hcn, original_key, cls_pooled_output, cls_output_matrix_nn, cls_score, \
        cls_score_matrix_nn, cls_score_matrix_nn_his1, cls_score_matrix_nn_his2, \
                match_layer_hidden_s, match_layer_attention_s, pair_senvec_top_s, final_seq_matriax_s \
            = self.encode(input_ids, attention_mask, token_type_ids, pairs_list, passage_length, pairs_num,
                          sep_positions, ground_truth, mask_cls, pairwise_labels,
                          sentence_input_id, sentence_attention_mask, sentence_length,
                          para_input_id, para_attention_mask, max_sentence_length,
                          cuda, head_mask)
        # 使用bart编码句子对，并进行层级attention，得到段落中每个句子的表示，再通过transformer编码段落中的句子

        num_sen = passage_length
        order = ground_truth
        target = order 
        tgt_len = num_sen
        batch, num = target.size()  # torch.Size([4, 5])

        tgt_len_less = tgt_len
        target_less = target

        target_mask = torch.zeros_like(target_less).byte()
        pointed_mask_by_target = torch.zeros_like(target).byte()

        ##### relative mask #####
        eye_mask = torch.eye(num).byte().cuda(int(cuda[-1])).unsqueeze(0)

        loss_left1_mask = torch.tril(target.new_ones(num, num), -1).unsqueeze(0).expand(batch, num, num)
        truth_left1 = loss_left1_mask - torch.tril(target.new_ones(num, num), -2).unsqueeze(0)
        rela_mask = torch.ones_like(truth_left1).byte().cuda(int(cuda[-1])) - eye_mask

        for b in range(batch):
            pointed_mask_by_target[b, :tgt_len[b]] = 1
            target_mask[b, :tgt_len_less[b]] = 1

            rela_mask[b, tgt_len[b]:] = 0
            rela_mask[b, :, tgt_len[b]:] = 0

        ####  decoder ####
        dec_inputs = document_matrix[torch.arange(document_matrix.size(0)).unsqueeze(1), target[:, :-1]]
        start = dec_inputs.new_zeros(batch, 1, dec_inputs.size(2))
        dec_inputs = torch.cat((start, dec_inputs), 1)

        ##### transformer decoder #####
        enc_dec_mask = pointed_mask_by_target.unsqueeze(1)

        # masked mask
        masked_mask = (1 - torch.triu(torch.ones(1, num, num), diagonal=1)).byte().cuda(int(cuda[-1]))
        dec_self_mask = enc_dec_mask & masked_mask

        dec_outputs = self.tran_decoder(dec_inputs, dec_self_mask, enc_output, enc_dec_mask)  # transformer解码器
        dec_outputs = dec_outputs * mask_cls[:, :, None].float()

        ####  relative vector and score  ####
        rela_vec_diret = self.rela_encode(cls_output_matrix_nn, cls_score_matrix_nn)

        pw_keys = []
        pointed_mask = [rela_mask.new_zeros(batch, 1, num)]

        for t in range(num):
            if t == 0:
                rela_mask = rela_mask.unsqueeze(-1)
                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)
            else:
                tar = target[:, t - 1]

                rela_mask[torch.arange(batch), tar] = 0
                rela_mask[torch.arange(batch), :, tar] = 0
                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)

                l1_mask[torch.arange(batch), tar, :] = 1

                if t > 1:
                    l2_mask[torch.arange(batch), target[:, t - 2], :] = 1

                pm = pointed_mask[-1].clone().detach()
                pm[torch.arange(batch), :, tar] = 1
                pointed_mask.append(pm)
            rela_vec_diret.masked_fill_(rela_mask == 0, 0)

            forw_pw = rela_vec_diret.mean(2)
            back_pw = rela_vec_diret.mean(1)

            pw_info = torch.cat((forw_pw, back_pw), -1)
            pw_key = self.pw_k(pw_info)
            pw_keys.append(pw_key.unsqueeze(1))

            dec_inp = dec_inputs[:, t:t + 1]

        # dec_outputs: 经过transformer解码器后的表示
        query = self.query_linear(dec_outputs).unsqueeze(2)

        original_key = original_key.unsqueeze(1)
        e = torch.tanh(query + original_key)

        e = self.tanh_linear(e).squeeze(-1)

        pointed_mask = torch.cat(pointed_mask, 1)
        pointed_mask_by_target = pointed_mask_by_target.unsqueeze(1).expand_as(pointed_mask)

        e.masked_fill_(pointed_mask == 1, -1e9)
        e.masked_fill_(pointed_mask_by_target == 0, -1e9)

        logp = F.log_softmax(e, dim=-1)
        logp = logp.view(-1, logp.size(-1))
        loss = self.critic(logp, target.contiguous().view(-1))

        target_mask = target_mask.view(-1)
        loss.masked_fill_(target_mask == 0, 0)

        ##### revised loss #####
        transform_loss = loss.reshape(batch, num)

        transform_loss_sample = torch.sum(transform_loss, -1)/(num_sen.float() + 1e-20 - 1)  
        new_original_loss = transform_loss_sample.sum()/batch

        loss = new_original_loss

        return loss


    def encode(self, input_ids, attention_mask=None, token_type_ids=None,
                pairs_list=None, passage_length=None, pairs_num=None, sep_positions=None, ground_truth=None, 
                mask_cls=None, pairwise_labels=None,
                sentence_input_id=None, sentence_attention_mask=None, sentence_length=None,
                para_input_id=None, para_attention_mask=None, max_sentence_length=None,
                cuda=None, head_mask=None):

        batch_size, max_pair_num, max_pair_len = input_ids.size()

        input_ids = input_ids.reshape(batch_size*max_pair_num, max_pair_len)
        attention_mask = attention_mask.reshape(batch_size*max_pair_num, max_pair_len)
        token_type_ids = token_type_ids.reshape(batch_size*max_pair_num, max_pair_len)

        _, max_num_sen, max_seq_length = sentence_input_id.size()  # (batch, max_sen_num, max_sen_len)
        sentence_input_id = sentence_input_id.view(batch_size * max_num_sen, -1)
        sentence_attention_mask = sentence_attention_mask.view(batch_size * max_num_sen, -1)

        outputs = self.model(pair_input_ids=input_ids,
                            pair_attention_mask=attention_mask,
                            sentence_input_id=sentence_input_id,
                            sentence_attention_mask=sentence_attention_mask,
                            sentence_length=sentence_length,
                            pairs_list=pairs_list,
                            passage_length=passage_length,
                            pairs_num=pairs_num,
                            para_input_id=para_input_id,
                            para_attention_mask=para_attention_mask,
                            max_sentence_length=max_sentence_length,
                            cuda=cuda,
                            use_cache=False,
                            )  # 经过bart的encoder部分得到结果

        top_vec_0 = outputs[0]  # bart最后一层的输出

        outputs_up2 = self.model_up2(pair_input_ids=input_ids,
                             pair_attention_mask=attention_mask,
                             sentence_input_id=sentence_input_id,
                             sentence_attention_mask=sentence_attention_mask,
                             sentence_length=sentence_length,
                             pairs_list=pairs_list,
                             passage_length=passage_length,
                             pairs_num=pairs_num,
                             para_input_id=para_input_id,
                             para_attention_mask=para_attention_mask,
                             max_sentence_length=max_sentence_length,
                             top_vec=top_vec_0,
                             cuda=cuda,
                             use_cache=False,
                             )  # 经过bart的encoder部分得到结果


        top_vec = outputs_up2[0]
        cls_pooled_output = outputs_up2[1]

        all_layer_hidden = outputs_up2[2]
        attention_score_all = outputs_up2[3]

        match_layer_hidden_stu = all_layer_hidden
        match_layer_attention_stu = attention_score_all

        # 层级attention
        pair_senvec_top, final_seq_matriax, cls_output_matrix_nn, cls_score, cls_score_matrix_nn, \
        cls_score_matrix_nn_his1, cls_score_matrix_nn_his2 \
            = self.two_level_encoder(top_vec, pairs_list, passage_length, pairs_num,
                                     sep_positions, cuda, mask_cls, cls_pooled_output)

        clean_sents_vec = final_seq_matriax * mask_cls[:, :, None].float()
        mask_cls_tranf = mask_cls.to(dtype=next(self.parameters()).dtype)
        # transformer编码器
        para_matrix = self.encoder(clean_sents_vec, mask_cls_tranf)

        clean_para_matrix = para_matrix * mask_cls[:, :, None].float()
        num_sen_newsize = (passage_length.float() + 1e-20).unsqueeze(1).expand(-1, self.hidden_size)
        para_vec = torch.sum(clean_para_matrix, 1)/num_sen_newsize

        hn = para_vec.unsqueeze(0) 
        cn = torch.zeros_like(hn)
        hcn = (hn, cn)

        keyinput = torch.cat((clean_sents_vec, clean_para_matrix), -1)
        key = self.key_linear(keyinput)

        return clean_sents_vec, clean_para_matrix, hcn, key, cls_pooled_output, cls_output_matrix_nn, cls_score, \
               cls_score_matrix_nn, cls_score_matrix_nn_his1, cls_score_matrix_nn_his2, \
               match_layer_hidden_stu, match_layer_attention_stu, pair_senvec_top, final_seq_matriax


    def step(self, dec_inputs, enc_output, original_keys, mask, rela_vec, rela_mask,
             hist_left1, hist_left2, l1_mask, l2_mask, cuda):

        current_step = dec_inputs.size(1)
        dec_self_mask = (1 - torch.triu(torch.ones(1, current_step, current_step), diagonal=1)).byte().cuda(int(cuda[-1]))
        dec_outputs = self.tran_decoder(dec_inputs, dec_self_mask, enc_output, None)

        query = dec_outputs[:, -1, :].unsqueeze(1)  #  (beam, hidden) (beam, 1, hidden)
        query = self.query_linear(query)

        rela_vec.masked_fill_(rela_mask.unsqueeze(-1) == 0, 0)
        forw_futu = rela_vec.mean(2)
        back_futu = rela_vec.mean(1)

        pw = torch.cat((forw_futu, back_futu), -1)

        e = torch.tanh(query + original_keys)
        e = self.tanh_linear(e).squeeze(2)

        e.masked_fill_(mask, -1e9)
        logp = F.log_softmax(e, dim=-1)

        return logp


def beam_search_pointer(args, model, input_ids, attention_mask=None, token_type_ids=None,
                pairs_list=None, passage_length=None, pairs_num=None, sep_positions=None, ground_truth=None, 
                mask_cls=None, pairwise_labels=None,
                sentence_input_id=None, sentence_attention_mask=None, sentence_length=None,
                para_input_id=None, para_attention_mask=None, max_sentence_length=None,
                cuda=None, head_mask=None):

    sentences, enc_output, dec_init, original_keys, cls_pooled_output, cls_output_matrix_nn, \
            cls_score, cls_score_matrix_nn, cls_score_matrix_nn_his1, cls_score_matrix_nn_his2, \
        match_layer_hidden_s, match_layer_attention_s, pair_senvec_top_s, final_seq_matriax_s             \
            = model.encode(input_ids, attention_mask, token_type_ids, pairs_list, passage_length, pairs_num,
            sep_positions, ground_truth, mask_cls, pairwise_labels,
                           sentence_input_id, sentence_attention_mask, sentence_length,
                           para_input_id, para_attention_mask, max_sentence_length,
                           cuda, head_mask)

    num_sen = passage_length
    sentences = sentences[:, :num_sen, :]
    original_keys = original_keys[:, :num_sen, :]
    document = sentences.squeeze(0)

    ### transformer decoder ###
    enc_output = enc_output[:, :num_sen, :]
    T, H = document.size()

    ######## relative decoder #######
    rela_vec = model.rela_encode(cls_output_matrix_nn, cls_score_matrix_nn)  # batch, n, n hidden+2

    ### history ###
    hist_left1, hist_left2 = model.history_encode(cls_output_matrix_nn, cls_score_matrix_nn, cls_score_matrix_nn)

    eye_mask = torch.eye(T).cuda(int(cuda[-1])).byte()
    eye_zeros = torch.ones_like(eye_mask) - eye_mask

    W = args.beam_size

    prev_beam = Beam(W)
    prev_beam.candidates = [[]]
    prev_beam.scores = [0]

    target_t = T - 1

    f_done = (lambda x: len(x) == target_t)

    valid_size = W
    hyp_list = []


    for t in range(target_t):
        candidates = prev_beam.candidates
        if t == 0:
            # start
            dec_input = sentences.new_zeros(1, 1, H)
            pointed_mask = sentences.new_zeros(1, T).byte()

            rela_mask = eye_zeros.unsqueeze(0).cuda(int(cuda[-1]))

            l1_mask = torch.zeros_like(rela_mask).cuda(int(cuda[-1]))
            l2_mask = torch.zeros_like(rela_mask).cuda(int(cuda[-1]))

        else:
            index = sentences.new_tensor(list(map(lambda cand: cand[-1], candidates))).long()

            cur_seq = sentences.new_tensor(candidates).long() 
            dec_input = document[cur_seq]


            temp_batch = index.size(0)

            pointed_mask[torch.arange(temp_batch), index] = 1

            rela_mask[torch.arange(temp_batch), :, index] = 0
            rela_mask[torch.arange(temp_batch), index] = 0

            l1_mask = torch.zeros_like(rela_mask).cuda(int(cuda[-1]))
            l2_mask = torch.zeros_like(rela_mask).cuda(int(cuda[-1]))

            l1_mask[torch.arange(temp_batch), index, :] = 1

            if t > 1:
                left2_index = index.new_tensor(list(map(lambda cand: cand[-2], candidates)))
                l2_mask[torch.arange(temp_batch), left2_index, :] = 1

        log_prob = model.step(dec_input, enc_output, original_keys, pointed_mask, rela_vec, rela_mask, 
            hist_left1, hist_left2, l1_mask, l2_mask, cuda)


        next_beam = Beam(valid_size)
        done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
        hyp_list.extend(done_list)
        valid_size -= len(done_list)

        if valid_size == 0:
            break

        beam_remain_ix = input_ids.new_tensor(remain_list).cuda(int(cuda[-1])) 

        pointed_mask = pointed_mask.index_select(0, beam_remain_ix)

        rela_mask = rela_mask.index_select(0, beam_remain_ix)
        rela_vec = rela_vec.index_select(0, beam_remain_ix)

        hist_left1 = hist_left1.index_select(0, beam_remain_ix)
        hist_left2 = hist_left2.index_select(0, beam_remain_ix)

        enc_output = enc_output.index_select(0, beam_remain_ix)

        prev_beam = next_beam


    score = enc_output.new_tensor([hyp[1] for hyp in hyp_list])
    
    sort_score, sort_ix = torch.sort(score)
    output = []
    for ix in sort_ix.tolist():
        output.append((hyp_list[ix][0], score[ix].item()))
    best_output = output[0][0]

    the_last = list(set(list(range(T))).difference(set(best_output)))
    best_output.append(the_last[0])

    return best_output






