"""Convert a vanilla tnl to tnl-moe"""
import os
import shutil
from collections import Counter

import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel
from smoe.models.tnl import (
    TransnormerConfig,
    TransnormerModel,
    TransnormerForCausalLM,
)

from smoe.models.tnl_moe import (
    TNLMoEConfig,
    TNLMoEModel,
    TNLMoEForCausalLM,
)
from smoe.utils.io import torch_load_template_file


def convert_tnl_model(
    tnl_model_path,
    split_index_path,
    select_gate_path,
    save_path,
    template,
    num_experts,
    num_selects,
    score_scale_factor=None,
    use_random_gate=False,
    gate_type="mlp",  # "linear"
    use_softmax=True,
    multiply_gate_scores=True,
):
    """
    TNLMoEModel
    """

    moe_indices = []
    moe_gates = []
    size_experts = []

    """load model"""
    print("Loading TransNormerLLM model...")
    model_tnl = TransnormerModel.from_pretrained(tnl_model_path)
    print("TransNormerLLM model loaded: ", model_tnl)
    model_tnl.to("cpu")
    model_tnl_state_dict = model_tnl.state_dict()

    """load indices and gate weights"""
    hidden_size = model_tnl.config.hidden_size
    num_layers = model_tnl.config.num_hidden_layers

    for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
        this_layer_index = torch_load_template_file(split_index_path, template, i)
        moe_indices.append(torch.tensor(this_layer_index, dtype=torch.int))

        this_layer_size_expert = Counter(this_layer_index)
        this_layer_size_expert = [this_layer_size_expert[j] for j in range(num_experts)]
        size_experts.append(this_layer_size_expert)

        if not use_random_gate:
            this_layer_gate = torch_load_template_file(select_gate_path, template, i)
            moe_gates.append(this_layer_gate)

    """build config"""
    print("Buiding tnl-moe config...")
    config_tnl_moe = TNLMoEConfig.from_pretrained(tnl_model_path)
    config_tnl_moe.num_experts = num_experts
    config_tnl_moe.num_selects = num_selects
    config_tnl_moe.size_experts = size_experts
    config_tnl_moe.gates = gate_type
    config_tnl_moe.gate_use_softmax = use_softmax
    config_tnl_moe.score_scale_factor = (
        1.0 if score_scale_factor is None else score_scale_factor
    )
    config_tnl_moe.multiply_gate_scores = multiply_gate_scores

    """initialize moe model"""
    print("Initializing tnl-moe model...")
    model_tnl_moe = TNLMoEModel(config_tnl_moe)
    print("tnl-moe model Initialized: ", model_tnl_moe)
    model_tnl_moe.to("cpu")
    model_tnl_moe_state_dict = model_tnl_moe.state_dict().copy()

    # fmt: off
    """conversion"""
    print("Locating state dict values...")
    for key in model_tnl_state_dict.keys():
        if "mlp" not in key:
            model_tnl_moe_state_dict[key] = model_tnl_state_dict[key].cpu().half()
        else:
            layer_index = int(key.split(".")[1])
            for expert_index in range(num_experts):
                if "gate" in key:
                    model_tnl_moe_state_dict["layers.{}.mlp.calculator.experts.weight_gate.{}".format(layer_index, expert_index)] = model_tnl_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "up" in key:
                    model_tnl_moe_state_dict["layers.{}.mlp.calculator.experts.weight_up.{}".format(layer_index, expert_index)] = model_tnl_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "down" in key:
                    model_tnl_moe_state_dict["layers.{}.mlp.calculator.experts.weight_down.{}".format(layer_index, expert_index)] = model_tnl_state_dict[key].transpose(0, 1)[moe_indices[layer_index] == expert_index].transpose(0, 1).cpu().half()

    for layer_index in range(num_layers):
        if not use_random_gate and gate_type == "mlp":
            model_tnl_moe_state_dict["layers.{}.mlp.gate.gate_network.0.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.0.weight"].cpu()
            model_tnl_moe_state_dict["layers.{}.mlp.gate.gate_network.2.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.2.weight"].cpu()
        model_tnl_moe_state_dict["layers.{}.mlp.gate.weight_noise.weight".format(layer_index)] = torch.zeros((num_experts, hidden_size), requires_grad=True)
    # fmt: on

    print("Converting...")
    model_tnl_moe.load_state_dict(model_tnl_moe_state_dict)
    model_tnl_moe = model_tnl_moe.half()

    """save to file"""
    if os.path.exists(save_path):
        print(f'Removed existed files in "{save_path}"')
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print("Saving converted model...")
    config_tnl_moe.save_pretrained(save_path)
    model_tnl_moe.save_pretrained(save_path)
    print(f'Converted TNLMoEModel saved to "{save_path}".')


def convert_tnl_model_for_causal_lm(
    tnl_model_path,
    split_index_path,
    select_gate_path,
    save_path,
    template,
    num_experts,
    num_selects,
    score_scale_factor=None,
    use_random_gate=False,
    gate_type="mlp",  # "linear"
    use_softmax=True,
    multiply_gate_scores=True,
):
    """
    TNLMoEForCausalLM
    """

    moe_indices = []
    moe_gates = []
    size_experts = []

    """load model"""
    print("Loading TransNormerLLM model...")
    model_tnl = TransnormerForCausalLM.from_pretrained(tnl_model_path, trust_remote_code=True)
    model_tnl.to("cpu")
    model_tnl_state_dict = model_tnl.state_dict()

    """load indices and gate weights"""
    hidden_size = model_tnl.config.hidden_dim
    num_layers = model_tnl.config.decoder_layers

    for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
        this_layer_index = torch_load_template_file(split_index_path, template, i)
        moe_indices.append(torch.tensor(this_layer_index, dtype=torch.int))

        this_layer_size_expert = Counter(this_layer_index)
        this_layer_size_expert = [this_layer_size_expert[j] for j in range(num_experts)]
        size_experts.append(this_layer_size_expert)

        if not use_random_gate:
            this_layer_gate = torch_load_template_file(select_gate_path, template, i)
            moe_gates.append(this_layer_gate)

    """build config"""
    print("Buiding tnl-moe config...")
    config_tnl_moe = TNLMoEConfig.from_pretrained(tnl_model_path)
    config_tnl_moe.num_experts = num_experts
    config_tnl_moe.num_selects = num_selects
    config_tnl_moe.size_experts = size_experts
    config_tnl_moe.gates = gate_type
    config_tnl_moe.gate_use_softmax = use_softmax
    config_tnl_moe.score_scale_factor = (
        1.0 if score_scale_factor is None else score_scale_factor
    )
    config_tnl_moe.multiply_gate_scores = multiply_gate_scores
    
    # add by weigao
    config_tnl_moe.intermediate_size = config_tnl_moe.hidden_dim
    config_tnl_moe.num_hidden_layers = config_tnl_moe.decoder_layers
    config_tnl_moe.hidden_size = config_tnl_moe.hidden_dim
    print(config_tnl_moe)

    """initialize moe model"""
    print("Initializing tnl-moe model...")
    model_tnl_moe = TNLMoEForCausalLM(config_tnl_moe)
    print(model_tnl_moe)
    model_tnl_moe.to("cpu")
    model_tnl_moe_state_dict = model_tnl_moe.state_dict().copy()

    # fmt: off
    """conversion"""
    print("Locating state dict values...")
    for key in model_tnl_state_dict.keys():
        if "mlp" not in key:
            model_tnl_moe_state_dict[key] = model_tnl_state_dict[key].cpu().half()
        else:
            layer_index = int(key.split(".")[2])
            for expert_index in range(num_experts):
                if "gate" in key:
                    model_tnl_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_gate.{}".format(layer_index, expert_index)] = model_tnl_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "up" in key:
                    model_tnl_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_up.{}".format(layer_index, expert_index)] = model_tnl_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "down" in key:
                    model_tnl_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_down.{}".format(layer_index, expert_index)] = model_tnl_state_dict[key].transpose(0, 1)[moe_indices[layer_index] == expert_index].transpose(0, 1).cpu().half()

    for layer_index in range(num_layers):
        if not use_random_gate and gate_type == "mlp":
            model_tnl_moe_state_dict["model.layers.{}.mlp.gate.gate_network.0.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.0.weight"].cpu()
            model_tnl_moe_state_dict["model.layers.{}.mlp.gate.gate_network.2.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.2.weight"].cpu()
        model_tnl_moe_state_dict["model.layers.{}.mlp.gate.weight_noise.weight".format(layer_index)] = torch.zeros((num_experts, hidden_size), requires_grad=True)
    # fmt: on

    print("Converting...")
    model_tnl_moe.load_state_dict(model_tnl_moe_state_dict)
    model_tnl_moe = model_tnl_moe.half()

    """save to file"""
    if os.path.exists(save_path):
        print(f'Removed existed files in "{save_path}"')
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print("Saving converted model...")
    config_tnl_moe.save_pretrained(save_path)
    model_tnl_moe.save_pretrained(save_path)
    print(f'Converted TNLMoEForCausalLM saved to "{save_path}".')


# def convert_tnl_model_for_sequence_classification(
#     tnl_model_path,
#     split_index_path,
#     select_gate_path,
#     save_path,
#     template,
#     num_experts,
#     num_selects,
#     score_scale_factor=None,
#     use_random_gate=False,
#     gate_type="mlp",  # "linear"
#     use_softmax=True,
#     multiply_gate_scores=True,
# ):
#     """
#     LlamaMoEForSequenceClassification
#     """

#     moe_indices = []
#     moe_gates = []
#     size_experts = []

#     """load model"""
#     print("Loading llama model...")
#     model_tnl = LlamaForSequenceClassification.from_pretrained(tnl_model_path)
#     model_tnl.to("cpu")
#     model_tnl_state_dict = model_tnl.state_dict()

#     """load indices and gate weights"""
#     hidden_size = model_tnl.config.hidden_size
#     num_layers = model_tnl.config.num_hidden_layers

#     for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
#         this_layer_index = torch_load_template_file(split_index_path, template, i)
#         moe_indices.append(torch.tensor(this_layer_index, dtype=torch.int))

#         this_layer_size_expert = Counter(this_layer_index)
#         this_layer_size_expert = [this_layer_size_expert[j] for j in range(num_experts)]
#         size_experts.append(this_layer_size_expert)

#         if not use_random_gate:
#             this_layer_gate = torch_load_template_file(select_gate_path, template, i)
#             moe_gates.append(this_layer_gate)

#     """build config"""
#     print("Buiding llama-moe config...")
#     config_tnl_moe = LlamaMoEConfig.from_pretrained(tnl_model_path)
#     config_tnl_moe.num_experts = num_experts
#     config_tnl_moe.num_selects = num_selects
#     config_tnl_moe.size_experts = size_experts
#     config_tnl_moe.gates = gate_type
#     config_tnl_moe.gate_use_softmax = use_softmax
#     config_tnl_moe.score_scale_factor = (
#         1.0 if score_scale_factor is None else score_scale_factor
#     )
#     config_tnl_moe.multiply_gate_scores = multiply_gate_scores

#     """initialize moe model"""
#     print("Initializing llama-moe model...")
#     model_tnl_moe = LlamaMoEForSequenceClassification(config_tnl_moe)
#     model_tnl_moe.to("cpu")
#     model_tnl_moe_state_dict = model_tnl_moe.state_dict().copy()

#     # fmt: off
#     """conversion"""
#     print("Locating state dict values...")
#     for key in model_tnl_state_dict.keys():
#         if "mlp" not in key:
#             model_tnl_moe_state_dict[key] = model_tnl_state_dict[key].cpu().half()
#         else:
#             layer_index = int(key.split(".")[2])
#             for expert_index in range(num_experts):
#                 if "gate" in key:
#                     model_tnl_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_gate.{}".format(layer_index, expert_index)] = model_tnl_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
#                 elif "up" in key:
#                     model_tnl_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_up.{}".format(layer_index, expert_index)] = model_tnl_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
#                 elif "down" in key:
#                     model_tnl_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_down.{}".format(layer_index, expert_index)] = model_tnl_state_dict[key].transpose(0, 1)[moe_indices[layer_index] == expert_index].transpose(0, 1).cpu().half()

#     for layer_index in range(num_layers):
#         if not use_random_gate and gate_type == "mlp":
#             model_tnl_moe_state_dict["model.layers.{}.mlp.gate.gate_network.0.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.0.weight"].cpu()
#             model_tnl_moe_state_dict["model.layers.{}.mlp.gate.gate_network.2.weight".format(layer_index)] = moe_gates[layer_index]["gate_network.2.weight"].cpu()
#         model_tnl_moe_state_dict["model.layers.{}.mlp.gate.weight_noise.weight".format(layer_index)] = torch.zeros((num_experts, hidden_size), requires_grad=True)
#     # fmt: on

#     print("Converting...")
#     model_tnl_moe.load_state_dict(model_tnl_moe_state_dict)
#     model_tnl_moe = model_tnl_moe.half()

#     """save to file"""
#     if os.path.exists(save_path):
#         print(f'Removed existed files in "{save_path}"')
#         shutil.rmtree(save_path)
#     os.makedirs(save_path)

#     print("Saving converted model...")
#     config_tnl_moe.save_pretrained(save_path)
#     model_tnl_moe.save_pretrained(save_path)
#     print(f'Converted LlamaMoEForSequenceClassification saved to "{save_path}".')


if __name__ == "__main__":
    tnl_model_path = ""
    split_index_path = ""  # split
    select_gate_path = ""  # select
    save_path = ""
    template = "layers.{}.mlp.gate_proj.weight"
    num_experts = 8
    num_selects = 2
    score_scale_factor = 8.0
    use_random_gate = False

    convert_tnl_model(
        tnl_model_path,
        split_index_path,
        select_gate_path,
        save_path,
        template,
        num_experts,
        num_selects,
        score_scale_factor=score_scale_factor,
        use_random_gate=use_random_gate,
    )

    # load test
    model_tnl_moe = TNLMoEForCausalLM.from_pretrained(save_path)
    print(model_tnl_moe)
