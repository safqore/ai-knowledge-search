import pdfplumber
import pandas as pd

def make_unique(headers):
    """Ensure column headers are unique by adding suffixes to duplicates."""
    seen = {}
    unique_headers = []
    for header in headers:
        if header in seen:
            seen[header] += 1
            unique_header = f"{header}_{seen[header]}"
        else:
            seen[header] = 0
            unique_header = header
        unique_headers.append(unique_header)
    return unique_headers

def merge_multilevel_headers(header_rows):
    """Merges headers spanning multiple rows by concatenating parts with a separator."""
    merged_headers = []
    for col_idx in range(len(header_rows[0])):  # Assuming consistent number of columns in header rows
        merged_header = " ".join([row[col_idx].strip() for row in header_rows if row[col_idx]])
        merged_headers.append(merged_header)
    return merged_headers

def extract_and_process_table(pdf_path, start_page, end_page):
    all_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(start_page, end_page):
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            
            for table in tables:
                # Assume the first few rows are headers; the remaining are data
                header_rows = table[:2]  # Adjust based on number of header rows
                data_rows = table[2:]  # Remaining rows are data

                # Merge multilevel headers
                merged_headers = merge_multilevel_headers(header_rows)
                unique_headers = make_unique(merged_headers)
                
                # Convert data to DataFrame with merged headers
                df = pd.DataFrame(data_rows, columns=unique_headers)
                all_data.append(df)
    
    # Combine all tables from pages
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# Example usage
pdf_path = "data/raw/national-cdf-list-v1.331.pdf"
df = extract_and_process_table(pdf_path, start_page=2, end_page=248)
print(df.head())

# Save to CSV
df.to_csv("data/processed/extracted_cdf_data.csv", index=False)