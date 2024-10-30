#From https://huggingface.co/distilbert/distilgpt2
#Pipeline is a faster approach to generation that removes the need to declare the tokenizer and model.

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='distilgpt2')
set_seed(7)

return_count = int(input("How many sequences would you like to generate: "))
user_input = input("Enter prompt: ")
results = generator(user_input, max_length=50, num_return_sequences = return_count)

for i, result in enumerate(results):
    print(f"Output {i+1}: {result['generated_text']}")
