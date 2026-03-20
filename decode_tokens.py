from transformers import AutoTokenizer

prompt = "It was dark and stormy"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
input_ids = tokenizer(prompt).input_ids

input_ids 

for t in input_ids:
    print("\t : ",tokenizer.decode(t))
    

