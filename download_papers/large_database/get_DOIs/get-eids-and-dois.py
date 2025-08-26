### script to consolidate a list of Scopus EIDs and DOIs of articles that are returned following a Scopus query

import pandas as pd
import requests
import math
import json
import os

#Sets the current working directory to be the same as the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

output_dict = {'EID': [], 'DOI': []}

key = 'PLACEHOLDER' # Please replace this string with your Elsevier TDM API key

# define query
query = ("(DOCTYPE(ar) OR DOCTYPE(cp)) AND ( TITLE-ABS-KEY ( "
        "( \"oxidative coupling of methane\" OR \"oxidative coupling "
        "of CH4\" OR \"oxidative condensation of methane\" OR \"oxidative "
        "condensation of CH4\" OR \"oxidative coupling methane\" OR "
        "\"oxidative coupling CH4\" OR \"oxidative condensation methane\" "
        "OR \"oxidative condensation CH4\" OR \"methane oxidative coupling\" "
        "OR \"CH4 oxidative coupling\" OR \"methane oxidative condensation\" "
        "OR \"CH4 oxidative condensation\" ) AND NOT ( \"non-oxidative\" "
        "OR nocm OR nonoxidative OR \"CO2 oxidative\") ) AND NOT TITLE "
        "( photocat* ) )")

url_search = 'https://api.elsevier.com/content/search/scopus?APIKey=' + key

# get the total # of search results
sr = requests.get(url=url_search,params={'query': query})
sj = json.loads(sr.text)
ntotal = int(sj['search-results']['opensearch:totalResults'])
npages = math.ceil(ntotal/200)

for p in range(npages):

    print(f'starting page {p}')

    body = {'query': query,
            'start': p*200,
            'count': 200
            }   

    sr = requests.get(url=url_search,params=body)
    #print(sr) # check response code
    sj = json.loads(sr.text)

    for idx in range(len(sj['search-results']['entry'])):
        output_dict['EID'].append(sj['search-results']['entry'][idx]['eid'])
        try:
            output_dict['DOI'].append(sj['search-results']['entry'][idx]['prism:doi'])
        except:
            output_dict['DOI'].append('no DOI found')

# write output CSV
output_df = pd.DataFrame(data=output_dict)
output_df.to_csv('EIDs-and-DOIs.csv', mode='a', header=True, index=False)
