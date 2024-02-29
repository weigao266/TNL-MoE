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
""" PyTorch TNL-MoE model."""
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

from ..tnl.modeling_transnormer import (
    TransnormerDecoderLayer,
    TransnormerPreTrainedModel,
    TransnormerModel,
    TransnormerForCausalLM,
)
from transformers.utils import ModelOutput, logging

from smoe.models.llama_moe.configuration_llama_moe import LlamaMoEConfig
from smoe.modules.moe.moe_layers import LinearGLUMoELayer, MoEMlpOutput
from smoe.modules.norm import WeightNorm

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

_CONFIG_FOR_DOC = "TNLMoEConfig"

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

@dataclass
class BaseMoEModelOutputWithPast(ModelOutput):
    """
    Args:
        num_dropped_tokens: layer idx to the number of dropped tokens
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    balance_loss: Optional[float] = None
    num_dropped_tokens: Optional[Tuple[torch.Tensor]] = None
    gate_load: Optional[Tuple[list]] = None
    gate_importance: Optional[Tuple[list]] = None


@dataclass
class MoECausalLMOutputWithPast(CausalLMOutputWithPast):
    balance_loss: Optional[float] = None
    num_dropped_tokens: Optional[Tuple[int]] = None
    gate_load: Optional[Tuple[list[torch.Tensor]]] = None
    gate_importance: Optional[Tuple[list[torch.Tensor]]] = None



class TNLMoEDecoderLayer(TransnormerDecoderLayer):

    def __init__(self, config: TNLMoEConfig, layer_index):
        super().__init__(config)

        gating_config = {
            # all gates
            "gate_type": config.gate_type,
            "gate_network": config.gate_network,
            "gate_use_softmax": config.gate_use_softmax,
            "gate_use_balance": config.gate_use_balance,
            "gate_balance_loss_weight": config.gate_balance_loss_weight,
            "gate_add_noise": config.gate_add_noise,
            # TopKBalancedNoisyGate
            "gate_noise_epsilon": config.gate_noise_epsilon,
        }
        calculator_config = {
            # all calculators
            "calculator_type": config.calculator_type,
            "multiply_gate_scores": config.multiply_gate_scores,
            "score_scale_factor": (
                config.score_scale_factor[layer_index]
                if isinstance(config.score_scale_factor, list)
                else config.score_scale_factor
            ),
            "add_weight_norm": config.add_weight_norm,
            # SwitchDropTokenCalculator
            "drop_tokens": config.drop_tokens,
            "dropped_padding": config.dropped_padding,
            "capacity_factor": config.capacity_factor,
        }

        self.channel_mixer = LinearGLUMoELayer(
            input_size=self.hidden_size,
            hidden_size=config.intermediate_size,
            output_size=self.hidden_size,
            hidden_act=config.hidden_act,
            num_experts=config.num_experts,
            num_selects=config.num_selects,
            size_experts=(
                config.size_experts[layer_index]
                if config.size_experts is not None
                else None
            ),
            bias=False,
            **gating_config,
            **calculator_config,
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
        # x = self.channel_mixer(x)
        # x = self.residual_connection(x, residual)
        moe_mlp_outputs: MoEMlpOutput = self.channel_mixer(x)
        x = self.residual_connection(moe_mlp_outputs.x, residual) #? moe_mlp_outputs.x

        # outputs = (x, )
        outputs = (
            x,
            moe_mlp_outputs.balance_loss,
            moe_mlp_outputs.num_dropped_tokens,
            moe_mlp_outputs.gate_load,
            moe_mlp_outputs.gate_importance,
        )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs

    def set_moe_num_selects(self, num_selects):
        self.mlp.set_num_selects(num_selects)

    def set_moe_gate_use_softmax(self, use_softmax):
        self.mlp.set_gate_use_softmax(use_softmax)

    def set_moe_gate_use_balance(self, use_balance):
        self.mlp.set_gate_use_balance(use_balance)

    def set_moe_gate_balance_loss_weight(self, balance_loss_weight):
        self.mlp.set_gate_balance_loss_weight(balance_loss_weight)

    def set_moe_gate_add_noise(self, add_noise):
        self.mlp.set_gate_add_noise(add_noise)

    def set_moe_gate_noise_epsilon(self, noise_epsilon):
        self.mlp.set_gate_noise_epsilon(noise_epsilon)

    def set_moe_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.mlp.set_calculator_multiply_gate_scores(multiply_gate_scores)

    def set_moe_calculator_score_scale_factor(self, score_scale_factor):
        self.mlp.set_calculator_score_scale_factor(score_scale_factor)

    def set_moe_calculator_drop_tokens(self, drop_tokens):
        self.mlp.set_calculator_drop_tokens(drop_tokens)

    def set_moe_calculator_dropped_padding(self, dropped_padding):
        self.mlp.set_calculator_dropped_padding(dropped_padding)

    def set_moe_calculator_capacity_factor(self, capacity_factor):
        self.mlp.set_calculator_capacity_factor(capacity_factor)

    def reset_gate_network(self):
        self.mlp.reset_gate_network()

    def reset_experts(self):
        self.mlp.reset_experts()

class TNLMoEPreTrainedModel(TransnormerPreTrainedModel):
    config_class = TNLMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TNLMoEDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, WeightNorm):
            module.reset_parameters()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TNLMoEModel):
            module.gradient_checkpointing = value

class TNLMoEModel(TransnormerModel, TNLMoEPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TMLMoEDecoderLayer`]

    Args:
        config: TNLMoEConfig
    """

    def __init__(self, config: TNLMoEConfig):
        super().__init__(config)

        self.layers = nn.ModuleList([])
        for i in range(config.decoder_layers):
            if len(self.linear_use_lrpe_list) > 0:
                config.linear_use_lrpe = self.linear_use_lrpe_list[i]
            self.layers.append(TNLMoEDecoderLayer(config))

        # Initialize weights and apply final processing
        self.post_init()


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

        # weigao: add for moe
        num_dropped_tokens = ()
        gate_load = ()
        gate_importance = ()

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
            if layer_outputs[1] is not None:
                balance_loss += layer_outputs[1]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[6 if output_attentions else 5], )

            if output_attentions:
                all_self_attns += (layer_outputs[5], )

            num_dropped_tokens += (layer_outputs[2],)
            gate_load += (layer_outputs[3],)
            gate_importance += (layer_outputs[4],)

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
        return BaseMoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            balance_loss=balance_loss,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            num_dropped_tokens=num_dropped_tokens,
            gate_load=gate_load,
            gate_importance=gate_importance,
        )

    def update_config(self):
        self.config.vocab_size = self.config.vocab_size
        self.config.max_position_embeddings = self.config.max_position_embeddings
        # ↓↓↓↓↓↓↓↓↓↓↓↓ changed here ↓↓↓↓↓↓↓↓↓↓↓↓ #
        self.config.hidden_size = self.layers[0].mlp.input_size
        self.config.intermediate_size = self.layers[0].mlp.hidden_size
        self.config.num_hidden_layers = len(self.layers)
        self.config.num_attention_heads = self.layers[0].self_attn.num_heads
        self.config.hidden_act = self.layers[0].mlp.hidden_act
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ #
        self.config.initializer_range = self.config.initializer_range
        self.config.rms_norm_eps = self.config.rms_norm_eps
        self.config.pretraining_tp = self.config.pretraining_tp
        self.config.use_cache = self.config.use_cache
        self.config.rope_scaling = self.config.rope_scaling
        self.config._rope_scaling_validation()

        self.config.num_experts = self.layers[0].mlp.num_experts
        self.config.num_selects = self.layers[0].mlp.num_selects
        self.config.size_experts = [
            self.layers[i].mlp.calculator.experts.size_experts
            for i in range(self.config.num_hidden_layers)
        ]

        self.config.gate_type = vars(self.layers[0].mlp).get(
            "gate_type", "TopKBalancedNoisyGate"
        )
        self.config.gate_network = vars(self.layers[0].mlp.gate).get(
            "gate_network_type", "mlp"
        )
        self.config.gate_use_softmax = vars(self.layers[0].mlp.gate).get(
            "use_softmax", True
        )
        self.config.gate_use_balance = vars(self.layers[0].mlp.gate).get(
            "use_balance", True
        )
        self.config.gate_balance_loss_weight = vars(self.layers[0].mlp.gate).get(
            "balance_loss_weight", 1e-2
        )
        self.config.gate_add_noise = vars(self.layers[0].mlp.gate).get(
            "add_noise", True
        )
        self.config.gate_noise_epsilon = vars(self.layers[0].mlp.gate).get(
            "noise_epsilon", 1e-2
        )

        self.config.calculator_type = vars(self.layers[0].mlp).get(
            "calculator_type", "UniversalCalculator"
        )
        self.config.multiply_gate_scores = vars(self.layers[0].mlp.calculator).get(
            "multiply_gate_scores", True
        )
        self.config.score_scale_factor = [
            vars(self.layers[i].mlp.calculator).get("score_scale_factor", 1.0)
            for i in range(self.config.num_hidden_layers)
        ]
        self.config.drop_tokens = vars(self.layers[0].mlp.calculator).get(
            "drop_tokens", True
        )
        self.config.dropped_padding = vars(self.layers[0].mlp.calculator).get(
            "dropped_padding", "zero"
        )
        self.config.capacity_factor = vars(self.layers[0].mlp.calculator).get(
            "capacity_factor", 1.25
        )

    def set_moe_num_selects(self, num_selects):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_num_selects(num_selects)

    def set_moe_gate_use_softmax(self, use_softmax):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_use_softmax(use_softmax)

    def set_moe_gate_use_balance(self, use_balance):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_use_balance(use_balance)

    def set_moe_gate_balance_loss_weight(self, balance_loss_weight):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_balance_loss_weight(balance_loss_weight)

    def set_moe_gate_add_noise(self, add_noise):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_add_noise(add_noise)

    def set_moe_gate_noise_epsilon(self, noise_epsilon):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_noise_epsilon(noise_epsilon)

    def set_moe_calculator_multiply_gate_scores(self, multiply_gate_scores):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_multiply_gate_scores(multiply_gate_scores)

    def set_moe_calculator_score_scale_factor(
        self, score_scale_factor, layer_index=None
    ):
        if layer_index is None:
            for idx, decoder_layer in enumerate(self.layers):
                decoder_layer.set_moe_calculator_score_scale_factor(score_scale_factor)
        else:
            self.layers[layer_index].set_moe_calculator_score_scale_factor(
                score_scale_factor
            )

    def set_moe_calculator_drop_tokens(self, drop_tokens):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_drop_tokens(drop_tokens)

    def set_moe_calculator_dropped_padding(self, dropped_padding):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_dropped_padding(dropped_padding)

    def set_moe_calculator_capacity_factor(self, capacity_factor):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_capacity_factor(capacity_factor)

    def reset_gate_network(self):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.reset_gate_network()

    def reset_experts(self):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.reset_experts()


class TNLMoEForCausalLM(TransnormerForCausalLM, TNLMoEPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = TNLMoEModel(config)
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
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseMoEModelOutputWithPast = self.model(
            input_ids=input_ids,
            attn_padding_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
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
            if outputs.balance_loss is not None and outputs.balance_loss > 0:
                loss += outputs.balance_loss

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            num_dropped_tokens=outputs.num_dropped_tokens,
            balance_loss=outputs.balance_loss,
            gate_load=outputs.gate_load,
            gate_importance=outputs.gate_importance,
        )

    def update_config(self):
        self.model.update_config()

    def set_moe_num_selects(self, num_selects):
        self.model.set_moe_num_selects(num_selects)

    def set_moe_gate_use_softmax(self, use_softmax):
        self.model.set_moe_gate_use_softmax(use_softmax)

    def set_moe_gate_use_balance(self, use_balance):
        self.model.set_moe_gate_use_balance(use_balance)

    def set_moe_gate_balance_loss_weight(self, balance_loss_weight):
        self.model.set_moe_gate_balance_loss_weight(balance_loss_weight)

    def set_moe_gate_add_noise(self, add_noise):
        self.model.set_moe_gate_add_noise(add_noise)

    def set_moe_gate_noise_epsilon(self, noise_epsilon):
        self.model.set_moe_gate_noise_epsilon(noise_epsilon)

    def set_moe_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.model.set_moe_calculator_multiply_gate_scores(multiply_gate_scores)

    def set_moe_calculator_score_scale_factor(
        self, score_scale_factor, layer_index=None
    ):
        self.model.set_moe_calculator_score_scale_factor(
            score_scale_factor, layer_index=layer_index
        )

    def set_moe_calculator_drop_tokens(self, drop_tokens):
        self.model.set_moe_calculator_drop_tokens(drop_tokens)

    def set_moe_calculator_dropped_padding(self, dropped_padding):
        self.model.set_moe_calculator_dropped_padding(dropped_padding)

    def set_moe_calculator_capacity_factor(self, capacity_factor):
        self.model.set_moe_calculator_capacity_factor(capacity_factor)

    def reset_gate_network(self):
        self.model.reset_gate_network()

    def reset_experts(self):
        self.model.reset_experts()


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
