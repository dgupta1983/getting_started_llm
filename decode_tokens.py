from transformers import AutoTokenizer

#prompt = "Hope this is not bringing asparagusohikdjjds in too many tokens as this is a very long sentence"
prompt="The quick brown fox jumps over the lazy dog."
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
tokenizer_gemma = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
input_ids = tokenizer(prompt).input_ids

input_ids 
#print("\t : ",input_ids)
token_count = 0
for t in input_ids:
    #print("\t : ",tokenizer.decode(t))
    token_count+=1
print("Number of words in the prompt : ",len(prompt.split()))
print("Number of tokens in the prompt : ",token_count)
input_ids_gemma = tokenizer_gemma(prompt).input_ids
print("\t Mistral: ",input_ids_gemma, "\t Qwen: ",input_ids)
token_count_gemma = 0
for t in input_ids_gemma:
    #print("\t : ",tokenizer_gemma.decode(t))
    token_count_gemma+=1
print("Number of words in the prompt : ",len(prompt.split()))
print("Number of tokens in the prompt : ",token_count_gemma)


    

