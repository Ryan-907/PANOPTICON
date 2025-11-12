from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PanoHandler import Panorama, PanoPlus
import torch

MODEL ='meta-llama/Meta-Llama-3.1-70B-Instruct'

#Accessing prompt
current_directory = Path(__file__).parent
prompt_gen_prompt = current_directory.parent / 'prompts' / 'synthetic_prompt_gen_prompt.md'

with open(prompt_gen_prompt, 'r', encoding='utf-8') as f:
    instruction = f.read()

#setting up model and generation
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map='auto', low_cpu_mem_usage=True, load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

gener = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, do_sample=True, temperature=0.6, top_p=0.9, max_new_tokens=300)

#Setting up datasets
pano = Panorama()
pano_p = PanoPlus()
ids = pano_p.get_ids() #list of unique IDs


#Main loop
for uid in ids[1:]:
    content = [] 

    for idx in range(len(ids)):

        if pano.articles['id'][idx] == uid: #finding unique ID of any article content type
            article = pano.articles['text'][idx]
            pii = list(pano_p.dataset[idx].items())[:35] #List of unique PII associated with profile
        if pano.socials['id'][idx] == uid:
            content.append(pano.socials['text'][idx])
        if pano.forums['id'][idx] == uid:
            content.append(pano.forums['text'][idx])
        if pano.reviews['id'][idx] == uid:
            content.append(pano.reviews['text'][idx])
        if pano.comments['id'][idx] == uid:
            content.append(pano.comments['text'][idx])
        if pano.ads['id'][idx] == uid:
            content.append(pano.ads['text'][idx])

    prompts = [{"role":"system", "content": f'{instruction}'}, {'role':'user', 'content': f'Article: {article}. Content: {content} PII List: {pii}'}]
   
    output = gener(prompts, truncation=True, max_length=300)
    print(output[0]['generated_text'])
    input()
    del output
    torch.cuda.empty_cache()
