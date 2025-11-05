from normalization_fns import normalize_names, normalize_percentages, normalize_temperatures
from dotenv import load_dotenv
import pandas as pd
import boto3
import os

# Set the current working directory to be the same as the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv('api.env')

# Read in the data
data = pd.read_csv('demo.csv')

# Create Bedrock client
client = boto3.client(service_name='bedrock-runtime', region_name=os.environ['AWS_REGION'])

# Normalize the property values
data = normalize_percentages(data, header='Property Value', min_value=0, max_value=100)

# Normalize the temperatures
data = normalize_temperatures(data, header='Operating Condition 1 Value', min_temp=0, max_temp=2000)

# Normalize the names
data = normalize_names(data, client, header='Material')

# Delete any rows where the Catalyst or Value columns became None
data = data[data['Material'] != 'None']
data = data[data['Property Value'] != 'None']

# Write the normalized data to a new CSV file
data.to_csv('demo-normalized.csv', index=False)
