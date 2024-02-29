#    Copyright 2024 OpenNLPLab
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# coding=utf-8
""" PyTorch Transnormer model."""
import math
import os
from typing import List, Optional, Tuple, Union

from einops import rearrange
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_transnormer import TransnormerConfig
from .norm import SimpleRMSNorm as SimpleRMSNorm_torch
from .srmsnorm_triton import SimpleRMSNorm as SimpleRMSNorm_triton
from .utils import (
    get_activation_fn,
    get_norm_fn,
    logging_info,
    print_module,
    print_params,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TransnormerConfig"

# TODO: fix environment: https://huggingface.co/OpenNLPLab/TransNormerLLM-7B/discussions/1
use_triton = eval(os.environ.get("use_triton", default="True"))
debug = eval(os.environ.get("debug", default="False"))
do_eval = eval(os.environ.get("do_eval", default="False"))
eval_and_not_generate = eval(
    os.environ.get("eval_and_not_generate", default="False"))
BLOCK = 256

if use_triton:
    try:
        from .lightning_attention2 import lightning_attention

        has_lightning_attention = True
    except (ImportError, ModuleNotFoundError):
        has_lightning_attention = False
else:
    has_lightning_attention = False

if debug:
    logger.info(f"Use triton: {use_triton}")
    logger.info(f"Use lightning attention: {has_lightning_attention}")
    logger.info(f"Debug mode: {debug}, {type(debug)}")

if not has_lightning_attention:

    def linear_attention(q, k, v, attn_mask):
        energy = torch.einsum("... n d, ... m d -> ... n m", q, k)
        energy = energy * attn_mask
        output = torch.einsum("... n m, ... m d -> ... n d", energy, v)

        return output


class TransnormerDecoderLayer(nn.Module):

    def __init__(self, config: TransnormerConfig):
        super().__init__()
        self.embed_dim = config.decoder_embed_dim
        ##### normalize
        norm_type = config.norm_type
        if debug:
            logging_info(f"Decoder Norm Type: {norm_type}")
        self.token_norm = get_norm_fn(norm_type)(self.embed_dim)
        self.channel_norm = get_norm_fn(norm_type)(self.embed_dim)

        ##### token mixer
        self.token_mixer = self.build_token_mixer(
            self.embed_dim,
            config,
        )

        ##### channel mixer
        self.glu_dim = config.glu_dim
        if self.glu_dim == -1:
            self.glu_dim = self.embed_dim
        bias = config.bias
        self.channel_mixer = GLU(self.embed_dim, self.glu_dim, bias)

    def build_token_mixer(self, embed_dim, config):
        return NormLinearAttention(
            embed_dim=embed_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.decoder_attention_heads,
            linear_act_fun=config.linear_act_fun,
            norm_type=config.norm_type,
            linear_use_lrpe=config.linear_use_lrpe,
            bias=config.bias,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
            self,
            x,
            attn_mask: Optional[torch.Tensor] = None,
            attn_padding_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            slope_rate: Optional[torch.Tensor] = None,  # (h, 1, 1)
    ):
        residual = x
        x = self.token_norm(x)
        x, self_attn_weights, present_key_value = self.token_mixer(
            x=x,
            attn_mask=attn_mask,
            attn_padding_mask=attn_padding_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            slope_rate=slope_rate,
        )
        x = self.residual_connection(x, residual)

        residual = x
        x = self.channel_norm(x)
        x = self.channel_mixer(x)
        x = self.residual_connection(x, residual)

        outputs = (x, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


TRANSNORMER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TransnormerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(TRANSNORMER_START_DOCSTRING, )
class TransnormerPreTrainedModel(PreTrainedModel):
    config_class = TransnormerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransnormerDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TransnormerModel):
            module.gradient_checkpointing = value


TRANSNORMER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attn_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attn_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(TRANSNORMER_START_DOCSTRING, )
class TransnormerModel(TransnormerPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TransnormerDecoderLayer`]

    Args:
        config: TransnormerConfig
    """

    def __init__(self, config: TransnormerConfig):
        super().__init__(config)
        # hf origin
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False
        # mask
        self._linear_attn_mask = torch.empty(0)
        # config
        self.linear_use_lrpe_list = config.linear_use_lrpe_list
        self.num_layers = config.decoder_layers
        # h, 1, 1
        self.slopes = self._build_slope_tensor(config.decoder_attention_heads)

        # params
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.decoder_embed_dim,
                                         self.padding_idx)
        self.layers = nn.ModuleList([])
        for i in range(config.decoder_layers):
            if len(self.linear_use_lrpe_list) > 0:
                config.linear_use_lrpe = self.linear_use_lrpe_list[i]
            self.layers.append(TransnormerDecoderLayer(config))

        self.final_norm = get_norm_fn(config.norm_type)(
            config.decoder_embed_dim)
        self.embed_dim = config.decoder_embed_dim
        self.embed_scale = (1.0 if config.no_scale_embedding else math.sqrt(
            self.embed_dim))

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):

        def get_slopes(n):

            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n
                )  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(
                    math.log2(n)
                )  # when the number of heads is not a power of 2, we use this workaround.
                return (get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                    2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        # h, 1, 1
        slopes = torch.tensor(get_slopes(n_attention_heads)).reshape(
            n_attention_heads, 1, 1)

        return slopes

    def extra_repr(self):
        return print_module(self)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_linear_attn_mask(self, input_shape, inputs_embeds,
                                          past_key_values_length):
        bsz, tgt_len = input_shape
        src_len = tgt_len + past_key_values_length

        def power_log(x):
            return 2**(math.ceil(math.log(x, 2)))

        n = power_log(max(tgt_len, src_len))
        if self._linear_attn_mask.shape[-1] < n:

            def get_mask(n):
                mask = torch.triu(
                    torch.zeros(n, n).float().fill_(float("-inf")), 1)
                # no slope version
                # -n, ..., -2, -1, 0
                for i in range(n):
                    x = torch.arange(i + 1)
                    y = x
                    mask[i, :i + 1] = -torch.flip(y, [0])

                return mask

            arr = []
            for slope in self.slopes:
                arr.append(get_mask(n))
            self._linear_attn_mask = torch.stack(arr, dim=0).to(inputs_embeds)

        linear_attn_mask = self._linear_attn_mask[:, -tgt_len:, -src_len:]
        num_heads = linear_attn_mask.shape[0]

        return linear_attn_mask[None, :, :, :].expand(bsz, num_heads, tgt_len,
                                                      src_len)

    @add_start_docstrings_to_model_forward(TRANSNORMER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attn_padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            # !!! use embed_scale
            inputs_embeds = self.embed_scale * self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        ##### norm linear layers
        linear_attn_padding_mask = attn_padding_mask
        linear_attn_mask = self._prepare_decoder_linear_attn_mask(
            (batch_size, seq_length), inputs_embeds, past_key_values_length)

        slope_rates = [
            self.slopes.to(input_ids.device) for _ in range(self.num_layers)
        ]

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            past_key_value = (past_key_values[idx]
                              if past_key_values is not None else None)

            slope_rate = slope_rates[idx]
            slope_rate = slope_rate * (1 - idx / (self.num_layers - 1) + 1e-5)
            mask = linear_attn_mask

            layer_outputs = layer(
                hidden_states,
                attn_mask=mask,
                attn_padding_mask=linear_attn_padding_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                slope_rate=slope_rate,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1], )

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        hidden_states = self.final_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class TransnormerForCausalLM(TransnormerPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = TransnormerModel(config)
        if debug:
            logging_info(self.model)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.decoder_embed_dim,
                                 config.vocab_size,
                                 bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(TRANSNORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast,
                               config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, TransnormerForCausalLM

        >>> model = TransnormerForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attn_padding_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past), )
        return reordered_past
