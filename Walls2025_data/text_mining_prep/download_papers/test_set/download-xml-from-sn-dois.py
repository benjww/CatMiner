import pandas as pd
import requests
import time
import os

#Sets the current working directory to be the same as the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define test set Springer Nature DOIs
dois_df = pd.read_csv('OA-and-pub-info.csv')
dois = dois_df[dois_df['Publisher'] == 'Springer Science and Business Media LLC']['DOI'].to_list()

key = 'PLACEHOLDER'

# Create the directory to save XML files in
if not os.path.exists('XMLs/Springer_Nature'):
    os.makedirs('XMLs/Springer_Nature')

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
    
    #time.sleep(0.1)
    # print(sr) # check response code

    # save as an XML file
    with open(f'XMLs/Springer_Nature/{doi[8:]}.xml', 'w', encoding='utf-8') as file:
        file.write(sr.text)

    print(f'Saved XML {i}')
