from random import sample, randint
import re
from ast import literal_eval

#Regex help
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

    p = re.sub(r'\([^)]*\)\s*$', '', p).strip()


    plus = p.startswith('+')
    digits = ''.join(ch for ch in p if ch.isdigit())
    if not digits:
        return ''
    return ('+' if plus else '') + digits

address_pattern = re.compile(r"[A-Za-z0-9']+")

child_map = {'1':'one','2':'two', '3':'three', '4':'four'}
#6 pools (+ benign handled elsewhere). All relevant PII attributes from PANORAMA (excluding Unique ID, Locale, and Complete Info as potential attributes)
professional_pii = ['First Name', 'Last Name', 'Work Email', 'Employer', 'Job Title', 'Annual Salary', 'Education Info', 'Work Phone', 'Address',  'Age'] #10
finance_pii = ['Annual Salary', 'Net Worth', 'Finance Status', 'Credit Score', 'Email Address', 'Spouse Name', 'Marital Status', 'Children Count', 'Work Phone', 'Employer'] #10
health_pii = ['Age', 'Gender', 'Blood Type', 'Allergies', 'Disability' ,'Emergency Contact Name', 'Emergency Contact Phone', 'Address', 'Birth Date', 'Children Count'] #10
relationship_pii = ['First Name', 'Last Name', 'Marital Status', 'Spouse Name', "Father's Name", "Mother's Name", "Passport Number", 'Children Count', 'Phone Number', 'Gender'] #10
government_pii = ['First Name', 'Last Name', 'National ID', 'Passport Number', 'Driver\'s License', 'Birth Date', 'Address', 'Phone Number', 'Nationality', 'Employer', 'Age'] #11
social_pii = ['Social Media Handles', 'First Name', 'Last Name', 'Spouse Name', 'Nationality', 'Email Address', 'Phone Number', "Father's Name", "Mother's Name", 'Age', 'Marital Status','Birth City', 'Education Info'] #13

pii_pools = [professional_pii, finance_pii, health_pii, relationship_pii, government_pii, social_pii]





def sample_pii(pool: int, full_pii: dict) -> dict[str: str]:
    potential_pii = sample(pii_pools[pool], randint(2,5)) #2-5 random PII attributes
    used_pii = {}
    for pii in potential_pii:
            used_pii[pii] = full_pii[pii]
    return used_pii


def match_pii(og_prompt: str, pii_values: dict) -> dict:

    pii = {}

    for k,v in pii_values.items():
        
        if k in ('First Name', 'Last Name', 'Gender', 'Nationality', 'Marital Status', 'Finance Status', 'Birth City', 'Disability', 'Employer', 'Education Info', 'Allergies', 'Job Title'):
            if v in re.findall(basic_text, og_prompt) or v.lower() in re.findall(basic_text, og_prompt):
                pii[k]=v

        elif k in ("Father's Name", "Mother's Name", 'Spouse Name', 'Emergency Contact Name'):
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
            if ('child' in og_prompt) or ('Child' in og_prompt) or ('children' in og_prompt) or ('Children' in og_prompt) or ('kid' in og_prompt) or ('Kid' in og_prompt) or ('Kids' in og_prompt) or ('kids' in og_prompt):
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
                if len(val.split()) < 2:
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

if __name__ == '__main__':
    #Testing
    pii = sample_pii(professional_pii, {'First Name': 'Walter', 'Last Name': 'White', 'Work Email':'walter.white@tntech.edu', 'Job Title': 'cook', 'Annual Salary': '$1,000,000', 'Education Info':'Master\'s', 'Work Phone':'123 456 789', 'Address':'308 Negra Arroyo Lane', 'Birth Date': '12/12/1980'})
    print(pii)