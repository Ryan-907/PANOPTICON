from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PanoHandler import Panorama, PanoPlus, match_pii
import torch

MODEL ="meta-llama/Llama-3.1-8B-Instruct"

#Accessing prompt
current_directory = Path(__file__).parent
system_prompt = current_directory.parent / 'prompts' / 'gen_editorial_prompt.md'

with open(system_prompt, 'r', encoding='utf-8') as f:
    instruction = f.read()

#setting up model and generation
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto', low_cpu_mem_usage=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, do_sample=True,temperature=.3, top_p=0.9, max_new_tokens=400)

#Setting up datasets
pano = Panorama()
pano_p = PanoPlus()
ids = pano_p.get_ids() #list of unique IDs


#Main loop
for uid in ids[]: #Skipping 1 as profile one is used to provide example for one shot in prompt
    #content = [] 

    #Filtering by uid to get appropriate information associated with each profile
    article = pano.articles.filter(lambda x:x['id'] == uid)
    #social = pano.socials.filter(lambda x:x['id'] == uid)
    #forums = pano.forums.filter(lambda x:x['id'] == uid)
    #reviews = pano.reviews.filter(lambda x:x['id'] == uid)
    #comments = pano.reviews.filter(lambda x:x['id'] == uid)
    #ads = pano.ads.filter(lambda x:x['id'] == uid)
    
    #Getting text data to pass
    article = article['text'][0] #Ensuring duplicate articles are not passed

    #Getting relevat PII
    full_pii = pano_p.dataset.filter(lambda x:x['Unique ID'] == uid)[0] #Ensuring duplicate PII is not passed
    pii = list(full_pii.items())[:35]


    #content.append(social['text'][:3]) #I am selecting three of each example because I like the way :3 looks
    #content.append(forums['text'][:3])
    #content.append(reviews['text'][:3])
    #content.append(comments['text'][:3])
    #content.append(ads['text'][:3])

    prompts = [{"role":"system", "content": f'{instruction}'}, {'role':'user', 'content': f'Article: {article} PII List: {pii}.'}]
   
    output = generator(prompts, truncation=True)
    output = output[0]['generated_text']

    print(output)

    pii = match_pii(output, full_pii)
    print(pii)
    input()

    del output
    torch.cuda.empty_cache()
