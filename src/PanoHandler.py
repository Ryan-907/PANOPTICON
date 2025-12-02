import datasets
from dataclasses import dataclass, field
from typing import List
import random
import re
from ast import literal_eval

basic_text = re.compile(r"[A-Za-z+-]+")
basic_number = re.compile(r'[0-9]+')
alphanumeric = re.compile(r"[A-Za-z0-9]+")
email = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
birthday = re.compile(
    r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
)
money = re.compile(r'[0-9$£€₹₱SR]+(?:\.\d+)')

phone_number = re.compile(r'\+?\d[\d\-\s\(\)]{4,}\d')
def normalize_phone(p: str) -> str:
    p = p.strip()
    # Drop trailing labels like "(Work)", "(Home)", etc.
    p = re.sub(r'\([^)]*\)\s*$', '', p).strip()

    # Keep leading '+' if present (for country code), then digits only
    plus = p.startswith('+')
    digits = ''.join(ch for ch in p if ch.isdigit())
    if not digits:
        return ''
    return ('+' if plus else '') + digits

address_pattern = re.compile(r"[A-Za-z0-9']+")

child_map = {'1':'one','2':'two', '3':'three', '4':'four'}

'''
Dealing with the following datasets:

 srirxml/PANORAMA
Features {
            id: String
            content-type: String
            text: string}

content-types [
                Article,
                Social Media
                Forum Post,
                Online Review,
                Blog/News Article Comment,
                Online Ad
                ]

Splits = ['train']
num rows = 384_794
Config names = ['default']

Example entry
    id: 4f7dc93c-14bf-4f47-bd89-84a36a7e1f62

    content-type: Social Media

    text: Raymond : Just wrapped up a productive morning 
    at WOBN Trust Inc. at 68 and still full of energy! 
    Engineering never stops 🚀 #EngineeringLife 
    #NeverTooOld


    
 Dataset: srirxml/PANORAMA-Plus
    Num of Features: 36
    Splits: ['train']
    rows: 9674
    configs: ['default']

    'en_US', 'en_GB', 'en_CA', 'en_AU', 'en_NZ', 
    'en_IE', 'en_IN', 'en_PH'
'''

@dataclass
class Panorama:
    name: str = 'srirxml/PANORAMA'
    features: List[str] = field(default_factory=lambda: ['id', 'content-type', 'text'])
    content_types: list[str] = field(default_factory=lambda: ['Article', 'Social Media', 'Forum Post', 'Online Review', 'Blog/News Article Comment', 'Online Ad'])
    split: str = 'train' 
    num_rows: int = 384_794 
    conf_name: str = 'default' 
    dataset: object = field(init=False)
    articles: list[dict] = field(init=False)
    socials: list[dict] = field(init=False)
    forums: list[dict] = field(init=False)
    reviews: list[dict] = field(init=False)
    comments: list[dict] = field(init=False)
    ads: list[dict] = field(init=False)
    

    def __post_init__(self):
        self.dataset = datasets.load_dataset(self.name, self.conf_name, split=self.split)
        self.articles = self.dataset.filter(lambda x:x['content-type'] == self.content_types[0])
        self.socials = self.dataset.filter(lambda x:x['content-type'] == self.content_types[1])
        self.forums = self.dataset.filter(lambda x:x['content-type'] == self.content_types[2])
        self.reviews = self.dataset.filter(lambda x:x['content-type'] == self.content_types[3])
        self.comments = self.dataset.filter(lambda x:x['content-type'] == self.content_types[4])
        self.ads = self.dataset.filter(lambda x:x['content-type'] == self.content_types[5])
    
@dataclass
class Entry:
    idx: int = None
    id: str = field(init=False)
    content_type: str = field(init=False)
    text: str = field(init=False)
    dataset: datasets = field(repr=None, default=Panorama().dataset)

    def __post_init__(self):
        self.id = self.dataset[self.idx]['id']
        self.content_type = self.dataset[self.idx]['content-type']
        self.text = self.dataset[self.idx]['text']

@dataclass
class PanoPlus:
    name: str = 'srirxml/PANORAMA-Plus'

    features: list[str] = field(default_factory=lambda: ['Unique ID', 'Locale', 'First Name', 'Last Name', 
                                                        "Father's Name", "Mother's Name", 'Gender', 'Age', 
                                                        'Nationality', 'Marital Status', 'Spouse Name', 
                                                        'Children Count', 'National ID', 'Passport Number', 
                                                        "Driver's License", 'Phone Number', 'Work Phone', 
                                                        'Address', 'Email Address', 'Work Email', 'Birth Date', 
                                                        'Birth City', 'Education Info', 'Finance Status', 
                                                        'Net Worth', 'Employer', 'Job Title', 'Annual Salary', 
                                                        'Credit Score', 'Social Media Handles', 'Blood Type', 
                                                        'Allergies', 'Disability', 'Emergency Contact Name', 
                                                        'Emergency Contact Phone', 'complete_info'])
    
    
    split: str = 'train'
    num_rows: int = 9_674
    conf_name: str = 'default'
    dataset: object = field(init=False)

    def __post_init__(self):
        self.dataset = datasets.load_dataset(self.name, self.conf_name, split = self.split)

    def get_ids(self):
        return self.dataset['Unique ID']
    
    def rand_features(self, features):
        num_features = random.randint(2,5)
        return random.sample(features, num_features)

def match_pii(og_prompt: str, pii_values: dict) -> dict:

    pii = {}

    for k,v in pii_values.items():
        #Base Case
        if v in og_prompt or v.lower() in og_prompt:
            pii[k] = v

        #Special Cases
        elif k in ('First Name', 'Last Name', 'Gender', 'Nationality', 'Marital Status', 'Finance Status'):
            if v in re.findall(basic_text, og_prompt) or v.lower() in re.findall(basic_text, og_prompt):
                pii[k]=v

        elif k in ("Father's Name", "Mother's Name", 'Spouse Name', 'Allergies', 'Emergency Contact Name', 'Job Title',  'Birth City', 'Disability', 'Education Info', 'Employer'):
            names = v.split()
            name = ''
            for n in names:
                if n in re.findall(basic_text, og_prompt) or n.lower() in re.findall(basic_text, og_prompt):
                    name+=f'{n} '
            name = name.rstrip()
            if name:
                pii[k]=name

        elif k in ('Age', 'Credit Score'):
            prompt = re.findall(basic_number, og_prompt)
            if v in prompt:
                pii[k]=v

        
        elif k == 'Children Count':
            if v == '0':
                pass
            elif 'child' in og_prompt or 'children' in og_prompt:
                pii[k]=v
            elif child_map[v] in og_prompt:
                pii[k]=v

        elif k in ('National ID', 'Passport Number', "Driver's License"):
            prompt = re.findall(alphanumeric, og_prompt)
            if v in prompt:
                pii[k]=v

        elif k in ('Phone Number', 'Work Phone', 'Emergency Contact Phone'):
            matches = re.findall(phone_number, og_prompt)
            normalized_matches = [normalize_phone(m) for m in matches]
            v_norm = normalize_phone(v)
            if v_norm and v_norm in normalized_matches:
                pii[k] = v

        elif k in ('Email Address', 'Work Email'):
            matches = re.findall(email, og_prompt)
            if v in matches:
                pii[k] = v

        elif k == 'Birth Date':
            prompt = re.findall(birthday, og_prompt)
            if v in prompt:
                pii[k]=v
        
        elif k in ("Annual Salary", "Net Worth"):
            prompt = og_prompt.replace(',','')
            prompt = re.findall(money, prompt)
            if v in prompt:
                pii[k] = v

        #Complex Cases
        elif k == 'Address':
            val = ''
            prompt = re.findall(address_pattern, og_prompt)
            check = re.findall(address_pattern, v)
            for i in check:
                if i in prompt:
                    val+=f'{i} '
            val = val.rstrip()
            if not val:
                pass
            else:
                pii[k]=val

        elif k == 'Social Media Handles':
            if pii_values[k] == '{}':
                pass

            else:
                handles = literal_eval(v)
                for platform, handle in handles.items():
                    if handle in og_prompt:
                        pii[platform] = handle
 
    return pii


if __name__=='__main__':

    #Datasets
    pano = Panorama()
    pano_p = PanoPlus()
    ids = pano_p.get_ids()


    #Main loop, gets unique profiles from the datasets
    for uid in ids:
        
        #Filtering PII by unique id, ensuring only 1 row is derived
        full_pii = pano_p.dataset.filter(lambda x:x['Unique ID'] == uid)[0]

        print(list(full_pii.items())[:35]) #Excluding 'complete info'

        pii = {k: full_pii[k] for k in full_pii}
        print('Matcing pii...')

        v = match_pii(og_prompt="68-year-old", pii_values=full_pii)
        print(v)
        input()

    print("Matcher works for exact matches more or less...")

                
