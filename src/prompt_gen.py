from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PanoHandler import Panorama, PanoPlus, match_pii
import torch
from random import sample

MODEL ="meta-llama/Llama-3.1-8B-Instruct"

#Accessing prompt
current_directory = Path(__file__).parent
system_prompt = current_directory.parent / 'prompts' / 'experimental.md'

with open(system_prompt, 'r', encoding='utf-8') as f:
    instruction = f.read()

#setting up model and generation
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto', low_cpu_mem_usage=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, do_sample=True,temperature=1.00, top_p=0.8, max_new_tokens=400)

#Setting up datasets
pano = Panorama()
pano_p = PanoPlus()
ids = pano_p.get_ids() #list of unique IDs


#Main loop
for uid in ids: #Skipping 1 as profile one is used to provide example for one shot in prompt
    data = []
    content = []

    #Filtering by uid to get appropriate information associated with each profile

    article = pano.articles.filter(lambda x:x['id'] == uid)

    social = pano.socials.filter(lambda x:x['id'] == uid)
    forums = pano.forums.filter(lambda x:x['id'] == uid)
    reviews = pano.reviews.filter(lambda x:x['id'] == uid)
    comments = pano.reviews.filter(lambda x:x['id'] == uid)
    ads = pano.ads.filter(lambda x:x['id'] == uid)

    data.append(social['text'])
    data.append(forums['text'])
    data.append(reviews['text'])
    data.append(comments['text'])
    data.append(ads['text'])

    for d in data:
        for post in d:
            content.append(d)

    content = sample(content, 10)


    #Getting text data to pass
    article = article['text'][0] #Ensuring duplicate articles are not passed

    #Getting relevat PII
    full_pii = pano_p.dataset.filter(lambda x:x['Unique ID'] == uid)[0] #Ensuring duplicate PII is not passed
    pii = list(full_pii.items())[:35]




    prompts = [{"role":"system", "content": f'{instruction}'}, {'role':'user', 'content': f'Article: {article} Content: {content}.  PII List: {pii}.'}]
   
    output = generator(prompts, truncation=True)
    output = output[0]['generated_text']

    print(output)

    pii = match_pii(output, full_pii)
    print(pii)
    input()

    del output
    torch.cuda.empty_cache()
