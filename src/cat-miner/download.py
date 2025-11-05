import pandas as pd
import requests
import json
import time
import os

# Function to check if open access
def _is_open_access(data):

    ### Inputs
    # data: paper data retrieved from CrossRef API in json format

    # Check for the 'license' field in the response
    if 'license' in data['message']:
        licenses = data['message']['license']
        
        for license_info in licenses:
            # Check if the license URL contains "creativecommons"
            if 'creativecommons' in license_info['URL']:
                return True
    
    return False


def get_pub_info(dois):
    
    ### Inputs
    # dois: DOIs to retrieve publisher and OA status for, using the CrossRef API [list]

    ### Outputs:
    # pub_info: the input DOIs ('DOI'), the associated publisher ('Publisher'), and the open-access status ('OA Status') [dict]

    # define output dictionary
    pub_info = {'DOI': [], 'Publisher': [], 'OA Status': []}

    print(f'Beginning to check {len(dois)} DOIs with CrossRef API...')

    for i in range(len(dois)):

        # define and record the DOI
        doi = dois[i]
        pub_info['DOI'].append(doi)

        # make the API call
        url_doc = f'https://api.crossref.org/works/{doi}'
        dr = requests.get(url=url_doc)
        #print(dr) # check response code
        try:
            dj = json.loads(dr.text)
        except Exception as e:
            pub_info['Publisher'].append('HTTP error')
            pub_info['OA Status'].append('HTTP error')
            continue

        # append publisher info
        try:
            pub_info['Publisher'].append(dj['message']['publisher'])
        except Exception as e:
            pub_info['Publisher'].append('no publisher found')

        # append OA status
        OA_status = _is_open_access(dj)
        pub_info['OA Status'].append(OA_status)
        
        print(f'appended DOI {i}')

    return pub_info


def get_Elsevier_XML(dois, key, delay=5, download_path='XMLs/Elsevier'):

    ### Inputs:
    # dois: a list of DOIs associated with Elsevier that we want to download XML copies of [list]
    # key: your personal Elsevier API key [str]
    # delay: the number of seconds to rest in between each API request (default: 5) [int]
    # download_path: the location where the XML file should be downloaded to (default: 'XMLs/Elsevier') [str]

    # define headers for Elsevier API request
    headers = {
            'X-ELS-APIKey': key,
            'Accept': 'text/xml'
            }

    # Create the directory to save XML files in
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    print(f'Beginning to save XML files for {len(dois)} Elsevier DOIs...')

    for i in range(len(dois)):

        # define individual DOI
        doi = dois[i]

        # define endpoint URL
        url_doc = 'https://api.elsevier.com/content/article/doi/' + doi

        # retrieve document information
        try: 
            sr = requests.get(url=url_doc,headers=headers)
        except Exception as e:
            print('Error:', e)
            continue

        #print(sr.text)

        time.sleep(delay)
        # print(sr) # Check response code

        # save as an XML file
        with open(f'{download_path}/{doi.replace("/","_")}.xml', 'w', encoding='utf-8') as file:
            file.write(sr.text)

        print(f'Saved XML {i}')


def get_SN_XML(dois, key, delay=5, download_path='XMLs/Springer_Nature'):

    ### Inputs:
    # dois: a list of DOIs associated with Elsevier that we want to download XML copies of [list]
    # key: your personal Elsevier API key [str]
    # delay: the number of seconds to rest in between each API request (default: 5) [int]
    # download_path: the location where the XML file should be downloaded to (default: 'XMLs/Springer_Nature') [str]

    # Create the directory to save XML files in
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for i in range(len(dois)):

        doi = dois[i]

        # define endpoint URL
        url_doc = 'https://api.springernature.com/openaccess/jats?q=doi:' + doi + f'&api_key={key}'

        # retrieve document information
        try:
            sr = requests.get(url=url_doc)
        except Exception as e:
            print('Error:', e)
            continue
        
        time.sleep(delay)
        # print(sr) # Check response code

        # save as an XML file
        with open(f'{download_path}/{doi[8:]}.xml', 'w', encoding='utf-8') as file:
            file.write(sr.text)

        print(f'Saved XML {i}')
