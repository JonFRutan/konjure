#From https://huggingface.co/distilbert/distilgpt2

import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_text = "Examples of character names: John Doe, Sarah Connor, Michael Smith. Now generate a realistic-sounding character name: "
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cpu")
attention_mask = tokenizer(input_text, return_tensors="pt", padding=True).attention_mask.to("cpu")

model = model.to("cpu")

output = model.generate(
    input_ids,
    do_sample = True,
    max_length=50,
    attention_mask=attention_mask,
    temperature=1.2,  # Increase temperature for more randomness
    top_p=0.9,
    num_beams=5,  # Beam search helps explore more options
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)


output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
