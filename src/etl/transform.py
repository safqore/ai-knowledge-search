# Transform /data/raw/national-cdf-list-v1.331.json by cleaning and transforming as follows:
    # JSON is a better representation of PDF tables hierarchical structure than CSV's flat structure
    # Repurpose the extract_and_process_table function to work with JSON files. Once done clean up the function and remove any unnecessary code
    # TODO: Obtain header names from .json file
    # TODO: Iterate through every row in .json file and correlate this with header rows (null rows will throw you off! Be careful)
    # TODO: Create exact JSON equivalent of PDF table
    # TODO: Clean \n in JSON file and ensure self descriptive
    # TODO: Workout a way to handle green colour coding in PDF (maybe use a separate column to denote green colour coding)
    # TODO: Save as /data/processed/national-cdf-list-v1.331.json for feature selection module
import json

# Load the raw JSON file
with open("data/raw/national-cdf-list-v1.331.json", "r") as file:
    data = json.load(file)

def process_set(data_set):
    """
    Process a single set of the raw data into the desired structure.
    """
    # Headers
    headers = data_set[1]
    
    # The first three important fields are in the same fixed columns
    ref_key = headers[0]
    drug_key = headers[1]
    indication_key = headers[2]
    criteria_key = headers[3]

    # Process the main entry
    result = {
        ref_key: data_set[2][0],
        drug_key: data_set[2][1],
        indication_key: data_set[2][2],
        criteria_key: []
    }

    # Collect "Criteria for use" and related rows
    for row in data_set[2:]:
        if row[3]:  # If there is content in the "Criteria for use" column
            result[criteria_key].append(row[3])

    # Process additional columns beyond "Criteria for use"
    other_columns = headers[4:]
    for i, header in enumerate(other_columns, start=4):
        values = [row[i] for row in data_set[2:] if row[i]]  # Collect non-empty values
        if len(values) == 1:  # If there is only one unique value
            result[header] = values[0]
        elif values:  # If there are multiple values
            result[header] = values

    return result

# Process all sets in the file
processed_data = [process_set(data_set) for data_set in data]

# Save the transformed data
with open("data/processed/processed_cdf_list.json", "w") as file:
    json.dump(processed_data, file, indent=4)

print("Transformation complete. Data saved to 'processed_cdf_list.json'.")