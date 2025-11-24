from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PanoHandler import Panorama, PanoPlus, match_pii
import torch
from random import sample, choice, randint
from jinja2 import Template

#Variation in scenarios for promp generation

scenarios = {
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
        "Study/work planning",
        "Productivity advice",
        "General writing tips"
    ]
}

professional_pii = ['First Name', 'Last Name', 'Work Email', 'Employer', 'Job Title', 'Annual Salary', 'Education Info', 'Work Phone', 'Address', 'Birth Date']
finance_pii = ['Annual Salary', 'Net Worth', 'Finance Status', 'Credit Score', 'Address','Email Address', 'Spouse Name', 'Marital Status', 'Children Count', 'Phone Number', 'Employer']
medical_pii = ['Age', 'Gender', 'Blood Type', 'Allergies', 'Disability' ,'Emergency Contact Name', 'Emergency Contact Phone', 'Address', 'Birth Date', 'Children Count']
relationship_pii = ['First Name', 'Last Name', 'Marital Status', 'Spouse Name', 'Children Count', 'Address', 'Phone Number']
government_pii = ['First Name', 'Last Name', 'National ID','Passport Number', 'Driver\'s License', 'Birth Date', 'Address', 'Phone Number', 'Email Address', 'Nationality', 'Employer']
social_pii = ['Social Media Handles', 'First Name', 'Last Name', 'Job Title', 'Nationality', 'Email Address']
benign = ['This is a benign prompt, do not use any PII']

pii_pools = [professional_pii, finance_pii, medical_pii, relationship_pii, government_pii, social_pii, benign]
MODEL ="meta-llama/Llama-3.1-8B-Instruct"

#Accessing prompt
current_directory = Path(__file__).parent
system_prompt = current_directory.parent / 'prompts' / 'experimental.md'
benign_system_prompt = current_directory.parent / 'prompts' / 'benign_prompt.md'

with open(system_prompt, 'r', encoding='utf-8') as f:
    instruction_md = f.read()

with open(benign_system_prompt, 'r', encoding='utf-8') as b:
    benign_instruction_md = b.read()

template = Template(instruction_md)

#setting up model and generation
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, do_sample=True,temperature=.8, top_p=0.8, max_new_tokens=200)


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
    content = sample(content, 10)


    #Getting text data to pass
    article = article['text'][0] #Ensuring duplicate articles are not passed

    #Getting relevat PII
    full_pii = pano_p.dataset.filter(lambda x:x['Unique ID'] == uid)[0] #Ensuring duplicate PII is not passed
    full_pii = list(full_pii.items())[:35]

    pool = 0
    for scenario in scenarios.keys():
        if scenario == 'Benign':
            print(f'{scenario}\n')
            instruction = benign_system_prompt
        else:
            pii = []
            pii_list = pii_pools[pool]
            for i in pii_list:
                for p in full_pii:
                    if i == p[0]:
                        pii.extend(p)

            print(f"PII List: {pii}\n")
            sub_scenario = choice(scenarios[scenario])
            print(f"subscenario: {sub_scenario}\n")
            instruction = template.render(category=scenario, subcategory=sub_scenario, pii=pii)
            pool+=1



        prompts = [{"role":"system", "content": f'{instruction}'}, {'role':'user', 'content': f'Article: {article} Content: {content}.'}]
   
        output = generator(prompts, truncation=True)
        output = output[0]['generated_text']

        print(output)

        matched_pii = match_pii(output, full_pii)
        print(matched_pii)

        del output
        torch.cuda.empty_cache()
    input()
