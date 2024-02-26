#!/usr/bin/bash

tnl_size="TNL-385M"

num_experts=4           #  4  8  16  32
num_selects=2            #  1  2  4  8
split_type=Random #  Graph-l1_norm  Graph-l2_norm  Clustering-l2  Clustering-cos  Random
proj_type=gate_proj      #  gate_proj  up_proj
select_type=positive     #  plain  positive  l1_norm  l2_norm

# use_random_gate="False" #  True  False
use_random_gate="True" #  True  False
gate_type="mlp"         #  mlp  linear
use_softmax="False"
multiply_gate_scores="False"

score_scale_factor=1.0 #  1.0  2.0  4.0  8.0  16.0
score_scale_factor_file_path=""
#score_scale_factor_file_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/mlp-layer-wise-scale-factors/llama_13B_dense

convert_type=TNLMoEForCausalLM #  TNLMoEModel  TNLMoEForCausalLM  TNLMoEForSequenceClassification

data_path=/cpfs01/user/sunweigao/TNL-MoE
model_path=/home/sunweigao/.cache/huggingface/hub/models--OpenNLPLab--TransNormerLLM-385M/snapshots/c78a6f3d9315be099c147429b263787bf79a0050
# save_path=/cpfs01/user/sunweigao/TNL-MoE/moefication_results/split
split_file_path=/cpfs01/user/sunweigao/TNL-MoE/moefication_results/split/c78a6f3d9315be099c147429b263787bf79a0050-4Expert-Split-Random

if [ ${use_random_gate} = "True" ]; then
  select_file_path=""
  save_path=${data_path}/models/${convert_type}/${split_type}/${tnl_size}-${num_experts}Select${num_selects}-${proj_type}-Scale${score_scale_factor}
else
  select_file_path="/mnt/petrelfs/share_data/quxiaoye/moefication_results/select/Clustering-l2/ReluLLaMA-7B-16Expert-Select-MLP-positive-random"
  save_path=${data_path}/models/${convert_type}/${split_type}-${select_type}/${tnl_size}-${num_experts}Select${num_selects}-${proj_type}-HardBCE
  #  select_file_path=${data_path}/moefication_results/select/${split_type}/${tnl_size}-${num_experts}Expert-Select-MLP-${select_type}
  #  save_path=${data_path}/models/${convert_type}/${split_type}-${select_type}/${tnl_size}-${num_experts}Select${num_selects}-${proj_type}
fi

# gpus=0
# cpus=8
# OMP_NUM_THREADS=2 srun --partition=MoE --job-name=convert --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \

python -m smoe.entrypoint.expert_construction.tnl_convert \
--model_path ${model_path} \
--split_file_path ${split_file_path} \
--select_file_path "${select_file_path}" \
--save_path ${save_path} \
--template layers.{}.mlp.${proj_type}.weight \
--num_experts ${num_experts} \
--num_selects ${num_selects} \
--use_random_gate ${use_random_gate} \
--gate_type ${gate_type} \
--use_softmax ${use_softmax} \
--multiply_gate_scores ${multiply_gate_scores} \
--score_scale_factor ${score_scale_factor} \
--score_scale_factor_file_path "${score_scale_factor_file_path}" \
--convert_type ${convert_type}
