from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PanoHandler import Panorama, PanoPlus, match_pii
import torch
from random import sample, choice
from jinja2 import Template
from datasets import Dataset
from time import time

entries = []
#Variation in scenarios for prompt generation. 7 Categories, 18 usable scenarios
CATEGORIES = {
    "Professional": [
        "Resume/CV",
        "HR communication",
        "Professional biography"
    ],

    "Finance": [
        "Budget planning",
        "Tax bracket/deduction inquiry",
        "Mortgage or loan application",
    ],

    "Health": [
        "Email to doctor's office",
        "Medical history summary",
        "Fitness/health plan",
    ],

    "Relationship": [
        "Message to partner/family member",
        "Relationship Advice",
        "Family trip planning",
    ],

    "Government": [
        "Passport renewal assistance",
        "Social services inquiry",
        "Identity verification request"
    ],

    "Digital Identity": [
        "Social media post",
        "Email help",
        "Advertise professional services"
    ],

    "Benign": [
        "nothing here"
    ]
}

professional_pii = ['First Name', 'Last Name', 'Work Email', 'Employer', 'Job Title', 'Annual Salary', 'Education Info', 'Work Phone', 'Address', 'Birth Date']
finance_pii = ['Annual Salary', 'Net Worth', 'Finance Status', 'Credit Score', 'Email Address', 'Spouse Name', 'Marital Status', 'Children Count', 'Work Phone', 'Employer']
medical_pii = ['Age', 'Gender', 'Blood Type', 'Allergies', 'Disability' ,'Emergency Contact Name', 'Emergency Contact Phone', 'Address', 'Birth Date', 'Children Count']
relationship_pii = ['First Name', 'Last Name', 'Marital Status', 'Spouse Name', "Father's Name", "Mother's Name", "Passport Number", 'Children Count', 'Phone Number']
government_pii = ['First Name', 'Last Name', 'National ID','Passport Number', 'Driver\'s License', 'Birth Date', 'Address', 'Phone Number', 'Email Address', 'Nationality', 'Employer']
social_pii = ['Social Media Handles', 'First Name', 'Last Name', 'Job Title', 'Nationality', 'Email Address']
benign = ['This is a benign prompt, do not use any PII']

pii_pools = [professional_pii, finance_pii, medical_pii, relationship_pii, government_pii, social_pii, benign]
MODEL ="meta-llama/Llama-3.1-8B-Instruct"

#Accessing prompt
current_directory = Path(__file__).parent
system_prompt = current_directory.parent / 'prompts' / 'gen_prompts.md'
benign_system_prompt = current_directory.parent / 'prompts' / 'benign_prompt.md'
response_system_prompt = current_directory.parent / 'prompts' / 'response.md'

with open(system_prompt, 'r', encoding='utf-8') as f:
    instruction_md = f.read()

with open(benign_system_prompt, 'r', encoding='utf-8') as b:
    benign_instruction_md = b.read()

'''with open(response_system_prompt, 'r', encoding='utf-8') as r:
    response_md = r.read()''' #Uncommenting this line, 153-156, and adding two columnds to the dict will enable response pairs

template = Template(instruction_md)

#setting up model and generation
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, do_sample=True,temperature=.4, top_p=0.9)


#Setting up datasets
pano = Panorama()
pano_p = PanoPlus()
ids = pano_p.get_ids() #list of unique IDs

def gen_content(uid):
    data =  []
    content = []
    social = pano.socials.filter(lambda x:x['id'] == uid)
    forums = pano.forums.filter(lambda x:x['id'] == uid)
    reviews = pano.reviews.filter(lambda x:x['id'] == uid)
    comments = pano.comments.filter(lambda x:x['id'] == uid)
    ads = pano.ads.filter(lambda x:x['id'] == uid)

    data.append(social['text'])
    data.append(forums['text'])
    data.append(reviews['text'])
    data.append(comments['text'])
    data.append(ads['text'])

    for d in data:
        for post in d:
            content.append(post)
    return sample(content, 10) #Getting 10 random sample writings for tone guidance

#Main loop
t1 = time()
for uid in ids[:50]: 

    #Getting article
    article = pano.articles.filter(lambda x:x['id'] == uid)
    article = article['text'][0] #Ensuring duplicate articles are not passed
    content = gen_content(uid)


    #Getting relevat PII
    full_pii = pano_p.dataset.filter(lambda x:x['Unique ID'] == uid)[0] #Ensuring duplicate PII is not passed
    l_pii = list(full_pii.items())[:35] #Listi of PII tuples for iteration and removing redundancy from "complete info"

    pool = 0
    for category in CATEGORIES.keys():
        if category == 'Benign':
            instruction = benign_instruction_md
            scenario = 'Benign'
        else:
            pii = []
            pii_list = pii_pools[pool]
            for i in pii_list:
                for p in l_pii:
                    if i == p[0]:
                        pii.extend(p) #List of unique PII associated with the uid

            scenario = choice(CATEGORIES[category]) #Selecting a scenario from list of categories
            instruction = template.render(category=category, scenario=scenario, pii=pii)
            pool+=1 #Next category



        prompts = [{"role":"system", "content": f'{instruction}'}, {'role':'user', 'content': f'Article: {article} Content: {content}.'}]

        #Generate prompt
        prompt = generator(prompts, truncation=True, max_new_tokens=200)
        prompt = prompt[0]['generated_text']

        #Generate LLM output
        #responses = [{'role':'system', 'content':f'{response_md}'}, {'role':'user', 'content':f'{prompt}'}]
        #response = generator(responses, truncation=True, max_new_tokens=600)
        #response = response[0]['generated_text']


        prompt_pii = match_pii(prompt, full_pii)
 
        entry = {'ID':uid,
                 'Category': category,
                 'Scenario': scenario,
                 'Prompt': prompt,
                 'PromptPII': prompt_pii,
                 'Content':content}

        entries.append(entry)
        del prompt

t2 = time()


elapsed = t2 - t1
per_uid = elapsed / len(ids[:50])

print(f"Total time for {len(ids[:50])} UIDs: {elapsed:.2f} seconds")
print(f"Average per UID: {per_uid:.2f} seconds")
print(f"Estimated time for 10k UIDs: {per_uid * 10000 / 3600:.2f} hours")

dataset = Dataset.from_list(entries)
dataset.save_to_disk("PANOPTICON")
dataset.to_json('PANOPTICON.JSON', lines=True)