import pandas as pd
import os

# Set the current working directory to be the same as the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read in the data
data = pd.read_csv('../full-ocm-database-normalized.csv')

# split the data into three new CSV files based on the Property column
yield_data = data[data['Property'] == 'C2(+) yield']
selectivity_data = data[data['Property'] == 'C2(+) selectivity']
conversion_data = data[data['Property'] == 'CH4 conversion']

# sort the rows by the Property Value column in descending order
yield_data = yield_data.sort_values(by='Property Value', ascending=False)
selectivity_data = selectivity_data.sort_values(by='Property Value', ascending=False)
conversion_data = conversion_data.sort_values(by='Property Value', ascending=False)

# save the data to new CSV files
yield_data.to_csv('C2_yield.csv', index=False)
selectivity_data.to_csv('C2_selectivity.csv', index=False)
conversion_data.to_csv('CH4_conversion.csv', index=False)
