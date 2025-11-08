from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PanoHandler import Panorama, PanoPlus

MODEL ='meta-llama/Llama-3.1-8B-Instruct'

#Accessing prompt
current_directory = Path(__file__).parent
prompt_gen_prompt = current_directory.parent / 'prompts' / 'synthetic_prompt_gen_prompt.md'

with open(prompt_gen_prompt, 'r', encoding='utf-8') as f:
    instruction = f.read()

#setting up model and generation
model = AutoModelForCausalLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
gener = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, max_new_tokens=2000, do_sample=False)

#Setting up datasets
pano = Panorama()
pano_p = PanoPlus()
ids = pano_p.get_ids() #list of unique IDs


#Main loop
for uid in ids:
    content = [] 

    for idx in range(len(ids)):

        if pano.articles['id'][idx] == uid: #finding unique ID of any article content type
            article = pano.articles['text'][idx]
            content.append(article)
            pii = list(pano_p.dataset[idx].items())[:35] #List of unique PII associated with profile
        if pano.socials['id'][idx] == uid:
            content.append(pano.socials['text'][idx])
            
    prompts = [{"role":"system", "content": f'{instruction}'}, {'role':'user', 'content': f'PANORAMA: {content} PANORAMA-PLUS: {pii}'}]
    output = gener(prompts)
    print(output[0]['generated_text'])
    print(content)
    input()
