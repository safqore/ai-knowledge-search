# Load step moves transformed data from a staging area into a target system, typically a data warehouse for analysis. 
# There is no such use case in this project, so the load step is not implemented.
import json
import pandas as pd

# Load the JSON data
with open('data/processed/processed_cdf_list.json', 'r') as f:
    data = json.load(f)

# Convert JSON to DataFrame
df = pd.json_normalize(data)
