from ast import literal_eval
import pandas as pd
import numpy as np
import os

# Set the current working directory to be the same as the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read in the data
data = pd.read_csv('../full-ocm-database-normalized.csv')

# define subset of data on desired property
property = 'C2(+) selectivity'
data = data[data['Property'] == property]

# get a list of unique catalysts
data['Material'] = data['Material'].apply(literal_eval)
catalysts = data['Material'].tolist()
unique_catalysts = set(map(frozenset, catalysts))
unique_catalysts = [set(list(unique_catalyst)) for unique_catalyst in unique_catalysts]

# read in list of all possible elements
elements = pd.read_csv('elements.csv')['Element'].tolist()

# define dictionary of candidate promoters, the change associated with their addition, and the standard error
promoter_performances = {'Element': [], 'Change': [], 'Standard Error': [], '# Examples': []}

for promoter in elements:
    
    promoted_catalysts = [catalyst for catalyst in unique_catalysts if promoter in catalyst]

    improvements = []

    for promoted_catalyst in promoted_catalysts:
        unpromoted_catalyst = promoted_catalyst-{promoter}
        if unpromoted_catalyst in unique_catalysts:

            #print(f'We found data on {promoted_catalyst} with and without the promoter')
            
            # compute the average property value for the promoted catalyst
            promoted_performance = np.mean(data[data['Material'].apply(lambda x: x == promoted_catalyst)]['Property Value'])
            #print(promoted_performance)

            # compute the average property value for the unpromoted catalyst
            unpromoted_performance = np.mean(data[data['Material'].apply(lambda x: x == unpromoted_catalyst)]['Property Value'])
            #print(unpromoted_performance)

            #print(f'Selectivity of {unpromoted_catalyst} changes by {promoted_performance-unpromoted_performance} when promoted with {promoter}')

            improvements.append(promoted_performance-unpromoted_performance)

    if len(improvements) > 0:
        #print(f'On average, {promoter} improves XXX by {np.mean(improvements)} (STDERR = {np.std(improvements)/np.sqrt(len(improvements))})')
        promoter_performances['Element'].append(promoter)
        promoter_performances['Change'].append(np.mean(improvements))
        promoter_performances['Standard Error'].append(np.std(improvements)/np.sqrt(len(improvements)))
        promoter_performances['# Examples'].append(len(improvements))


# write to a CSV file
pd.DataFrame(promoter_performances).to_csv('selectivity_promoters.csv', index=False)
