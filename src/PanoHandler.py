import datasets
from dataclasses import dataclass, field
from typing import List
import random
import re
from ast import literal_eval

default_pattern = re.compile(r"[A-Za-z0-9'()_\/\\+-]+")
num_pattern = re.compile(r'[0-9$]+')
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

        if k in ['Education Info', 'Phone Number', "Work Phone", "Emergency Contact Phone", "Address", "Email Address", "Work Email", "Employer"]:
            if v in og_prompt:
                pii[k] = v

        elif k in ["Annual Salary", "Net Worth"]:
            prompt = re.findall(num_pattern, og_prompt)
            if v in prompt:
                pii[k] = v

        elif k in ['Social Media Handles']:
            if pii_values[k] == '{}':
                pass

            else:
                entry = ''
                handles =literal_eval(v)
                for sk, h in handles.items():
                    if h in og_prompt:
                        key = str(sk)
                        value = str(h)
                        entry += f"'{key}': '{value}', "
                entry = list(entry)

                try:
                    entry.pop(-1)
                    entry.pop(-1)
                    entry = ''.join(entry)
                    entry = "{"+entry+"}"
                    pii[k]=entry

                except:
                    continue

        elif k in ['Age']:
            prompt = re.findall(num_pattern, og_prompt)
            if v in prompt:
                pii[k]=v

        elif k in ['Children Count']:

            if v == 0:
                pass

            elif 'child' in prompt:
                pii[k]=v

            elif child_map[v] in prompt:
                pii[k]=v
            
        else:
            prompt = re.findall(default_pattern, og_prompt) 
            s = v.split()
            if len(s) > 1:
                if (s[0] in prompt and s[1] in prompt) or (s[0].lower() in prompt and s[1].lower() in prompt):
                    pii[k] = v
                elif s[0] in prompt or s[0].lower() in prompt:
                    pii[k] = s[0]
                elif s[1] in prompt or s[1].lower() in prompt:
                    pii[k] = s[1]
                else:
                    continue
            else:
                if v in prompt or v.lower() in prompt:
                    pii[k]= v
 
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

        v = match_pii(og_prompt="As I approach my 68th birthday, I've been reflecting on my career as a mechanical engineer at WOBN Trust Inc. and the various challenges I've faced in my role. With a PhD in hand, I've always strived to bring a unique blend of technical expertise and creative problem-solving to the table, often navigating complex engineering challenges with precision and innovation. My experience working under financial constraints has taught me the value of pragmatism, and I've learned to approach each project with a clear-eyed focus on what's feasible and what's not. I'm proud of the work I've done, from designing and implementing new systems to mentoring junior engineers, and I'm eager to continue making meaningful contributions to the field. Now, I'm looking to update my résumé to better reflect my accomplishments and experience, and I'd love some help from a skilled editor to ensure it accurately showcases my skills and achievements.", pii_values=full_pii)
        print(v)

        input()

    print("Matcher works for exact matches more or less...")

                
