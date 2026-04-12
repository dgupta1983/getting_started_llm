from transformers import AutoTokenizer

prompt = "Hope this is not bringing asparagusohikdjjds in too many tokens as this is a very long sentence"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
input_ids = tokenizer(prompt).input_ids

input_ids 
print("\t : ",input_ids)
token_count = 0
for t in input_ids:
    print("\t : ",tokenizer.decode(t))
    token_count+=1
print("Number of words in the prompt : ",len(prompt.split()))
print("Number of tokens in the prompt : ",token_count)
    
    

