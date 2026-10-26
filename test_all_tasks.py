import os
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
    set_seed
)

print("Starting verification of core logic...")
# 1. DistilBERT pipeline test
clf = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
res = clf("This framework is incredibly intuitive.")
print("1.1 pipeline:", res)

# 2. DistilBERT tokenizer
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
tokens = tok.tokenize("Transformers are surprisingly easy to understand.")
print("1.4 tokens:", tokens)

# 2.6 manual vs model attention test
model = AutoModel.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english', attn_implementation='eager')
inputs = tok("Transformers are powerful.", return_tensors='pt')
outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
print("DistilBERT attention shape:", outputs.attentions[0].shape)

# Test T5
t5_tok = AutoTokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small', attn_implementation='eager')
enc = t5_tok("translate English to French: The house is small.", return_tensors='pt')
dec = t5_tok("La maison est petite.", return_tensors='pt')
t5_out = t5_model(input_ids=enc.input_ids, labels=dec.input_ids, output_attentions=True)
print("T5 cross attention shape:", t5_out.cross_attentions[-1].shape)

# Test GPT-2
gpt2_tok = AutoTokenizer.from_pretrained('gpt2')
gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2', attn_implementation='eager')
gpt2_in = gpt2_tok("Artificial intelligence will", return_tensors='pt')
gpt2_out = gpt2_model(**gpt2_in, output_attentions=True)
print("GPT-2 attention shape:", gpt2_out.attentions[-1].shape)

print("All pre-checks passed!")
