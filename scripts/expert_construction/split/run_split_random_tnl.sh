#!/usr/bin/bash

tnl_size="TNL-385M"

num_experts=4 #  8  16

# data_path=/cpfs01/user/sunweigao/TNL-MoE
model_path=/home/sunweigao/.cache/huggingface/hub/models--OpenNLPLab--TransNormerLLM-385M/snapshots/c78a6f3d9315be099c147429b263787bf79a0050
save_path=/cpfs01/user/sunweigao/TNL-MoE/moefication_results/split

# gpus=0
# cpus=8
# OMP_NUM_THREADS=2 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \

python -m smoe.entrypoint.expert_construction.tnl_split_random \
--model_path ${model_path} \
--save_path ${save_path} \
--template layers.{}.mlp.gate_proj.weight \
--num_experts ${num_experts}
