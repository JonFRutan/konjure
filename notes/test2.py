#From https://huggingface.co/google/flan-t5-small
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
#This is a test using HF transformers for generation against a ML sklearn.

#tokenizer (specifically for T5) breaks out input into token IDS for the model.
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")
#Conditional Generation means accepting a command; then producing an output based on it.
#device_map = "auto" | This places the model on available devices, so when we call cuda it can use our GPU.
 
input_text = "Generate a list of 10 common names"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
#The tokenizer takes our input and creates 

output = model.generate(input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True)) #skip_special_tokens=True
