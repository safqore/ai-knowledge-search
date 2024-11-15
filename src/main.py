import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extraction.data_extraction import extract_and_process_table
#from src.query.query_pinecone import query_pinecone

def main():
    # Example usage of data extraction
    pdf_path = "data/raw/national-cdf-list-v1.331.pdf"
    df = extract_and_process_table(pdf_path, start_page=2, end_page=248)
    print(df.head())

    # Example usage of querying Pinecone
    #query_pinecone()

if __name__ == "__main__":
    main()