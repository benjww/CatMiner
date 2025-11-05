from parsing_fns import elsevier_to_sentences, rsc_to_sentences
import os

#Sets the current working directory to be the same as the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create the directory to save sentence files in
if not os.path.exists('sentences'):
    os.makedirs('sentences')

# Convert files from markup language into text file inputs for CatMiner
rsc_to_sentences('RSC_HTMLs/', 'sentences/')
elsevier_to_sentences('Elsevier_XMLs/', 'sentences/')

# delete the intermediate files that were generated
for file in os.listdir('sentences'):
    if 'continuous' in file:
        os.remove(os.path.join('sentences', file))
