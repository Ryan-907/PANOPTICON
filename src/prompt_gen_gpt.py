from pathlib import Path
from random import sample
from openai import OpenAI
from PanoHandler import Panorama, PanoPlus, match_pii
import torch

# === OpenAI / GPT setup ===
# Make sure OPENAI_API_KEY is set in your environment.
client = OpenAI()

GPT_MODEL = "gpt-5-chat-latest" 


# Accessing prompt
current_directory = Path(__file__).parent
system_prompt = current_directory.parent / 'prompts' / 'experimental.md'

with open(system_prompt, 'r', encoding='utf-8') as f:
    instruction = f.read()

# Setting up datasets
pano = Panorama()
pano_p = PanoPlus()
ids = pano_p.get_ids()  # list of unique IDs


def generate_with_gpt(messages):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.3,
        top_p=0.9,
        max_tokens=400,
    )
    return response.choices[0].message.content


# Main loop
for uid in ids:  # Skipping 1 previously? your original code didn't actually skip it
    data = []
    content = []

    # Filtering by uid to get appropriate information associated with each profile
    article = pano.articles.filter(lambda x: x['id'] == uid)
    social = pano.socials.filter(lambda x: x['id'] == uid)
    forums = pano.forums.filter(lambda x: x['id'] == uid)
    reviews = pano.reviews.filter(lambda x: x['id'] == uid)
    comments = pano.reviews.filter(lambda x: x['id'] == uid)
    ads = pano.ads.filter(lambda x: x['id'] == uid)

    data.append(social['text'])
    data.append(forums['text'])
    data.append(reviews['text'])
    data.append(comments['text'])
    data.append(ads['text'])

    for d in data:
        for post in d:
            content.append(d)

    content = sample(content, 10)

    # Getting text data to pass
    article = article['text'][0]  # Ensuring duplicate articles are not passed

    # Getting relevant PII
    full_pii = pano_p.dataset.filter(lambda x: x['Unique ID'] == uid)[0]  # Ensuring duplicate PII is not passed
    pii = list(full_pii.items())[:35]

    prompts = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": f"Article: {article} Content: {content} PII List: {pii}.",
        },
    ]

    # Call GPT instead of local Llama pipeline
    output = generate_with_gpt(prompts)

    print(output)

    pii = match_pii(output, full_pii)
    print(pii)
    input()

    del output
    torch.cuda.empty_cache()
