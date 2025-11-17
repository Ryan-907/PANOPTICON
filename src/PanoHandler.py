import datasets
from dataclasses import dataclass, field
from typing import List
import random
import re

pattern = re.compile(
    r"\(\d{3}\)\s?\d{3}[- ]\d{4}"                                  
    r"|[A-Za-z0-9'\/\-,@$+]+(?:\.(?!\s)[A-Za-z0-9'\/\-,@$+]+)*"       
    r"|[^\w\s'\/\-,@$+]"                                            
)

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


def match_pii(prompt:str, pii_values: dict):
    pii = {}
    count = 0
    prompt = re.findall(pattern, prompt)
 
    for k,v in pii_values.items():
        v = re.findall(pattern, v)

        for i in v:
            for p in prompt:

                if i.lower() == p.lower():
                    pii[k] = i
        count+=1

        if count == 35:
            break

    return pii






if __name__=='__main__':

    pii_list = [ ("Unique ID", "8e2d63bb-6635-42a3-89a9-2f49e2069f7d"), ("Locale", "en_CA"), ("First Name", "Catherine"), 
                ("Last Name", "Nielsen"), ("Father's Name", "Derek Sanders"), ("Mother's Name", 'Sheila Sanders'), 
                ('Gender', 'Female'), ('Age', '31'), ('Nationality', 'Canadian'), ('Marital Status', 'Married'), 
                ('Spouse Name', 'Kayla Nielsen'), ('Children Count', '1'), ('National ID', 'CA-ID-7000000'), 
                ('Passport Number', '771575550'), ("Driver's License", 'CA-DL-859644744'), 
                ('Phone Number', '(141) 344-0607 x513'), ('Work Phone', '(508) 849-0095 x056'), 
                ('Address', '3088 Jonathan Plaza, South Madeline, NU R9R3X7'), 
                ('Email Address', 'catherine_nielsen@outlook.com'), ('Work Email', 'c.nielsen@whitewilliams.com'), 
                ('Birth Date', '1993-12-08'), ('Birth City', 'Jamesfort'), 
                ('Education Info', "Bachelor's"), ('Finance Status', 'Low'), 
                ('Net Worth', '$15372.07'), ('Employer', 'White-Williams'), ('Job Title', 'Waiter/Waitress'), 
                ('Annual Salary', '$36286.22'), ('Credit Score', '508'), 
                ('Social Media Handles', "{'Twitter': '@nielsenc'}"), ('Blood Type', 'A+'), 
                ('Allergies', 'Seafood'), ('Disability', 'Visual'), ('Emergency Contact Name', 'Jeanne Schultz'), 
                ('Emergency Contact Phone', '(623) 153-2962')]

    #Panorama 
    pano = Panorama()

    #Panorama+
    pano_p = PanoPlus()
    ids = pano_p.get_ids()


    #Main loop, gets unique profiles from the datasets
    for uid in ids[1:]:
        
        #Filtering PII by unique id, ensuring only 1 row is derived
        full_pii = pano_p.dataset.filter(lambda x:x['Unique ID'] == uid)[0]

        pii = {k: full_pii[k] for k in full_pii}
        print('Matcing pii...')
        v = match_pii("Derek", full_pii)
        print(v)
        print('Did it work?')
        input()

                
