import json
import pandas as pd

# Load the JSON data
with open('data/processed/processed_cdf_list.json', 'r') as f:
    data = json.load(f)

# Convert JSON to DataFrame
df = pd.json_normalize(data)