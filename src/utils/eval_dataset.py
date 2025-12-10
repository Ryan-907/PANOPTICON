from datasets import load_from_disk  
import matplotlib.pyplot as plt
import numpy as np
from random import sample

dataset = load_from_disk("PANOPTICON\PANOPTICON_v1.2")

def handle_duplicates(dataset):
    # Identify duplicates in the dataset
    uniques = [] 
    duplicates = []
    duplicate_dict = {}
    categories_dict = {}

    # Iterate through dataset to find duplicates
    for x, i in enumerate(dataset):
        print(f'Duplicate handle: {((x/67718)*100):.2f}%', end='\r')
        if i['Prompt'] not in uniques:
            uniques.append(i['Prompt'])

        else:
            duplicates.append(i['Prompt'])

            if i['Prompt'] not in duplicate_dict.keys():
                duplicate_dict[i['Prompt']] = 1
            else:
                duplicate_dict[i['Prompt']] += 1
            
            if i['Category'] not in categories_dict.keys():
                categories_dict[i['Category']] = 1
            else:
                categories_dict[i['Category']] += 1

    # Print summary statistics
    print(f"Unique Prompts: {len(uniques)}")
    print('---------------------')
    print(f"Duplicate Prompts: {len(duplicates)}")
    print('---------------------')
    print(f'Single Dupes: {len(set(duplicates))}')
    print('---------------------')
    print(f'Most Duplicated Prompt: {max(duplicate_dict, key=duplicate_dict.get)}')
    print('---------------------')
    print(f'Most Duplicated Prompt Count: {max(duplicate_dict.values())}')
    print('---------------------')
    print(f'Categories with Duplicates: {categories_dict.items()}')
    print('---------------------')
    print(f'Full duplicate dict: {duplicate_dict} ')
    print('---------------------')

    # Return the results
    return uniques, duplicates, duplicate_dict, categories_dict

def handle_pii_lables(dataset):
    # Potential PII labels
    labels = {'Address': 0, 'Age': 0, 'Allergies': 0, 'Annual Salary': 0, 
              'Birth City': 0, 'Birth Date': 0, 'Blood Type': 0, 
              'Children Count': 0, 'Credit Score': 0, 'Disability': 0, 
              "Driver's License": 0, 'Education Info': 0, 'Email Address': 0, 
              'Emergency Contact Name': 0, 'Emergency Contact Phone': 0, 
              'Employer': 0, 'Facebook': 0, "Father's Name": 0, 
              'Finance Status': 0, 'First Name': 0, 'Gender': 0, 
              'Instagram': 0, 'Job Title': 0, 
              'Last Name': 0, 'LinkedIn': 0, 'Locale': 0, 'Marital Status': 0, 
              "Mother's Name": 0, 'National ID': 0, 'Nationality': 0, 'Net Worth': 0, 
              'Passport Number': 0, 'Phone Number': 0, 'Pinterest': 0, 'Reddit': 0, 
              'Snapchat': 0, 'Spouse Name': 0, 'TikTok': 0, 'Twitter': 0, 'Work Email': 0, 
              'Work Phone': 0, 'Unique ID': 0}
    
    # Iterate through dataset to count PII labels
    for x, i in enumerate(dataset):
        print(f'PII Label Handle: {x/67718}', end='\r')
        for label in i['PromptPII'].keys():

            if i['PromptPII'][label] != None:
                labels[label] += 1
                if label == 'Unique ID':
                    #print(i)
                    pass

    print(f'PII Label Counts: {labels}')
    sorted_items = sorted(labels.items(), key=lambda x: x[1], reverse=True)

    # Unpack into labels + values
    labels = [k for k, v in sorted_items]
    values = [v for k, v in sorted_items]

    # Graphic info
    plt.figure(figsize=(14, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    
    # Graph data
    plt.bar(labels, values, color=colors)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Graph labels
    plt.xlabel('PII Labels')
    plt.ylabel('Counts')
    plt.title('PII Label Distribution')
    plt.show()

    return labels


if __name__ == "__main__":  

    uniques, duplicates, duplicate_dict, categories_dict = handle_duplicates(dataset)
    #labels = handle_pii_lables(dataset)



