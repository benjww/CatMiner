from extract import catminer_abbreviation_resolution as catminer
from functions import define_client
from dotenv import load_dotenv
import pandas as pd
import os

# set the current working directory to be the same as the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# specify if we're using Meta-type or OpenAI-type models
model_type = 'Meta'

# load environment file
try:
    if load_dotenv('api.env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()

# create the client
client_type = 'Bedrock'
client = define_client(client_type)

# initialize total token count
total_in = 0
total_out = 0

# define input file paths
source_dir = 'OCM-papers/'
filenames = sorted(os.listdir(source_dir))
sp_paths = ['OCM-system-prompt-yield.txt', 'OCM-system-prompt-selectivity.txt', 'OCM-system-prompt-conversion.txt']

# define output file paths
log_path = 'log.csv'
record_path = "records.csv"

# define target properties
prop_names = ['C2(+) yield', 'C2(+) selectivity', 'CH4 conversion']

# define target reaction conditions (in this case we have two), 
# number of sentences that should be appended to search for them (including article title), 
# and phrases that must be contained within a sentence for CatMiner to search it in inter-paragraph mode (optional)
conditions = [{'Name': 'reaction operating temperature', 'Context Params': {'Bounds': [6, 0], 'Title': False}, 'Required Phrases': [' K', '°C']},
              {'Name': 'gas hourly space velocity', 'Context Params': {'Bounds': [6, 0], 'Title': False}, 'Required Phrases': ['gas', 'hourly', 'space', 'velocity', 'GHSV']}]

# begin extraction one paper at a time
print(f'Beginning extraction from {len(filenames)} papers.')
for p in range(len(filenames)):

    file_path = source_dir + filenames[p]

    print(f'Extracting no. {p}, {filenames[p]}...')

    # for each parent property...
    for i in range(len(prop_names)):

        # Compile target data, excerpt sizes, and required phrases 
        # NOTE if you have no required phrases, please send a blank list (i.e., 'Required Phrases': [''])
        properties = [{'Name': f'{prop_names[i]}', 'Context Params': {'Bounds': [2, 0], 'Title': True}, 'Required Phrases': ['%']}]
        target_dict = {'Properties': properties, 'Operating Conditions': conditions}
        print(f"Property to extract: {target_dict['Properties'][0]['Name']}.")

        # iterate through the input files
        sp_path = sp_paths[i]
        
        # call CatMiner
        catminer_output, cm_in_tkn, cm_out_tkn = catminer(file_path, client, target_dict, model_type, 
                                                        sp_path=sp_path, log_path=log_path, log_bool=True, 
                                                        SYSPROMPT=True, FOLLOWUP=[3], IPS=True, CHAT=True)
        print(f'Extracted {prop_names[i]} from {filenames[p]}. Input tokens: {cm_in_tkn}, Output tokens: {cm_out_tkn}')

        # count tokens
        total_in += cm_in_tkn
        total_out += cm_out_tkn

        # write CatMiner output to a CSV
        records_df = pd.DataFrame(data=catminer_output)
        if p == 0:
            records_df.to_csv(record_path, mode='a', index=False)
        else:
            records_df.to_csv(record_path, mode='a', header=False, index=False)

    print(f'Total input tokens so far: {total_in}. Total output tokens: {total_out}.')
