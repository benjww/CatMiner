### script to get the publisher and open-access status of a list of DOIs

import pandas as pd
import requests
import json
import time
import os

# define function to check open access
def is_open_access(data):

    # Check for the 'license' field in the response
    if 'license' in data['message']:
        licenses = data['message']['license']
        
        for license_info in licenses:
            # Check if the license URL contains "creativecommons"
            if 'creativecommons' in license_info['URL']:
                return True
    
    return False

#Sets the current working directory to be the same as the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# define output dictionary
output_dict = {'DOI': [], 'Publisher': [], 'OA Status': []}

# read input DOI list
dois_df = pd.read_csv('all_dois.csv')
dois_list = dois_df['DOI'].to_list()

print(f'Beginning to check {len(dois_list)} DOIs with CrossRef API...')

for i in range(len(dois_list)):

    # define and record the DOI
    doi = dois_list[i]
    print(i)
    output_dict['DOI'].append(doi)

    # make the API call
    url_doc = f'https://api.crossref.org/works/{doi}'
    dr = requests.get(url=url_doc)
    #print(dr) # check response code
    try:
        dj = json.loads(dr.text)
    except Exception as e:
        output_dict['Publisher'].append('HTTP error')
        output_dict['OA Status'].append('HTTP error')
        continue

    # append publisher info
    try:
        output_dict['Publisher'].append(dj['message']['publisher'])
    except Exception as e:
        output_dict['Publisher'].append('no publisher found')

    # append OA status
    OA_status = is_open_access(dj)
    output_dict['OA Status'].append(OA_status)
    
    print(f'appended DOI {i}')

# write output CSV
output_df = pd.DataFrame(data=output_dict)
output_df.to_csv('OA-and-pub-info.csv', mode='a', header=True, index=False)
