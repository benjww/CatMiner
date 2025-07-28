from ast import literal_eval
import pandas as pd
import os

# Set the current working directory to be the same as the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read in the data
data = pd.read_csv('../full-ocm-database-normalized.csv')

# literally evaluate the catalysts column
data['Material'] = data['Material'].apply(literal_eval)

# remove 'O' from every catalyst
data['Material'] = data['Material'].apply(lambda x: set([element for element in x if element != 'O']))

# get a list of the literally evaluated catalysts
catalysts = data['Material'].tolist()

# determine number of unique catalysts
unique_catalysts = set(map(frozenset, catalysts))

# unfreeze the unique catalysts
unique_catalysts = [set(list(unique_catalyst)) for unique_catalyst in unique_catalysts]

# Define dictionary of unique catalysts and the number of unique sources that investigate it
unique_catalysts_dict = {'Catalyst': [], 'N Sources': []}

# for each unique catalyst, print the number of uique sources that investigate it
for unique_catalyst in unique_catalysts:
    nsources = len(set(data[data['Material'].apply(lambda x: x == unique_catalyst)]['Source']))
    unique_catalysts_dict['Catalyst'].append(unique_catalyst)
    unique_catalysts_dict['N Sources'].append(nsources)

# save the dictionary to a CSV file
pd.DataFrame(unique_catalysts_dict).to_csv('unique_catalyst_sources.csv', index=False, header=False)
