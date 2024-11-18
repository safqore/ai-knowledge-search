# Transform /data/raw/national-cdf-list-v1.331.json by cleaning and transforming as follows:
    # JSON is a better representation of PDF tables hierarchical structure than CSV's flat structure
    # Repurpose the extract_and_process_table function to work with JSON files. Once done clean up the function and remove any unnecessary code
    # TODO: Obtain header names from .json file
    # TODO: Iterate through every row in .json file and correlate this with header rows (null rows will through you off! Be careful)
    # TODO: Create exact JSON equivalent of PDF table
    # TODO: Clean \n in JSON file and ensure self descriptive
    # TODO: Workout a way to handle green colour coding in PDF (maybe use a separate column to denote green colour coding)
    # TODO: Save as /data/processed/national-cdf-list-v1.331.json for feature selection module