import pandas as pd
import os

#Sets the current working directory to be the same as the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OA_and_pub_info = pd.read_csv('OA-and-pub-info.csv')

pubs = OA_and_pub_info['Publisher'].to_list()
dois = OA_and_pub_info['DOI'].to_list()
elsevier_dois = []

for i, pub in enumerate(pubs):
    if (pub == 'Elsevier') or (pub == 'Elsevier BV'):
        elsevier_dois.append(dois[i])

# Write to a new CSV file
elsevier_dois_df = pd.DataFrame({'DOI': elsevier_dois})
elsevier_dois_df.to_csv('elsevier-dois.csv', index=False)
