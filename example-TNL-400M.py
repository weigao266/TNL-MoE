# python>=3.10

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "OpenNLPLab/TransNormerLLM-385M"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
# )
model = AutoModelForCausalLM.from_pretrained(
    model_dir, trust_remote_code=True
)
model.eval()
model.to("cuda:0")
print(model)

input_text = "The United States of America (USA or U.S.A.), commonly known as the United"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda:0")

# pred = model.generate(**inputs, max_length=50, temperature=0.0)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three
