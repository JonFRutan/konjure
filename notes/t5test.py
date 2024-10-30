#From https://huggingface.co/google/flan-t5-small
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
#This is a test using HF transformers for generation against a ML sklearn.

#tokenizer (specifically for T5) breaks out input into token IDS for the model.
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")
#Conditional Generation means accepting a command; then producing an output based on it.
#device_map = "auto" | This places the model on available devices, so when we call cuda it can use our GPU.
 
input_text = "What are five numbers between 1 and 100?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
#The tokenizer takes our input and creates 

model_generation = model.generate(
    input_ids, 
    max_new_tokens=100 
    #temperature=0.7, 
    #top_p=0.9, 
    #num_beams=5, 
    #repetition_penalty=1.2
)

model_output = tokenizer.decode(model_generation[0], skip_special_tokens=True)
print("CUDA Available: " + str(torch.cuda.is_available()))
print(model_output)


#I am having trouble getting this model to produce anything valid-
#For now I am going to move on and try DistilGPT-2, perhaps I'll return to this if I realize
#I have been making in error here.
