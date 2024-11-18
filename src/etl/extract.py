import pdfplumber
import json

# Extraction only, no transformation or cleaning
def pdf_to_json(pdf_path, start_page, end_page, json_path):
    data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(start_page, end_page):
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            for table in tables:
                data.append(table)
    
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)