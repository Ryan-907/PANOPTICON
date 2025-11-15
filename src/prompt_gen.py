from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PanoHandler import Panorama, PanoPlus
import torch

MODEL ="meta-llama/Llama-3.1-8B-Instruct"

#Accessing prompt
current_directory = Path(__file__).parent
prompt_gen_prompt = current_directory.parent / 'prompts' / 'experimental.md'

with open(prompt_gen_prompt, 'r', encoding='utf-8') as f:
    instruction = f.read()

#setting up model and generation
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto', low_cpu_mem_usage=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

gener = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, do_sample=True,temperature=.3, top_p=0.9, max_new_tokens=300)

#Setting up datasets
pano = Panorama()
pano_p = PanoPlus()
ids = pano_p.get_ids() #list of unique IDs


#Main loop
for uid in ids[1:]: #Skipping 1 as profile one is used to provide example for one shot in prompt
    content = [] 
    #Filtering PII by unique id, ensuring only 1 row is derived
    full_pii = pano_p.dataset.filter(lambda x:x['Unique ID'] == uid)[0]

    #Randomly selecting PII features to use, getting key value pairs
    features = pano_p.rand_features(pano_p.ResumeFeatures)
    pii = {k: full_pii[k] for k in features}

    #Removing null PII
    pii = {k: pii[k] for k in pii if pii[k]!='{}' and pii[k]!= 'N/A'}
        
    #Handles edge case of no PII in dict
    if not pii:
        features = ['First Name', 'Last Name']
        pii = {k: full_pii[k] for k in features} 

    #Filtering by uid to get appropriate information associated with each profile
    article = pano.articles.filter(lambda x:x['id'] == uid)
    
    social = pano.socials.filter(lambda x:x['id'] == uid)
    forums = pano.forums.filter(lambda x:x['id'] == uid)
    reviews = pano.reviews.filter(lambda x:x['id'] == uid)
    comments = pano.reviews.filter(lambda x:x['id'] == uid)
    ads = pano.ads.filter(lambda x:x['id'] == uid)
    
    #Getting text data to pass
    article = article['text'][0] #Ensuring duplicate articles are not passed

    social = social['text'][:3] #I am selecting three of each example because I like the way :3 looks
    forums = forums['text'][:3]
    reviews = reviews['text'][:3]
    comments = comments['text'][:3]
    ads = ads['text'][:3]

    content.append(social)
    content.append(forums)
    content.append(reviews)
    content.append(comments)
    content.append(ads)

    prompts = [{"role":"system", "content": f'{instruction}'}, {'role':'user', 'content': f'Article: {article}. Content: {content} PII List: {pii}'}]
   
    output = gener(prompts, truncation=True)

    print(output[0]['generated_text'])
    print(pii)
    input()
    del output
    torch.cuda.empty_cache()
