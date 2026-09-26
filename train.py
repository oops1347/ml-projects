import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen3-14B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
for name, param in model.named_parameters():
    print(name, param.shape)


