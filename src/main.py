import sys
import os

# Add the parent directory to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_extraction.data_extraction import extract_and_process_table

"""
Stages of machine learning projects:

    1. Collection of the data
        Data available in /data/raw/

    2. Cleaning and transformation
        clean_and_transform_data()

    3. Feature selection
        TBD: 
            We will select a subset of input features from the data for a model to reduce noise. 
            We eliminate some of the available features in this process to get the best results from the model using minimum data and to ensure model explainability and simplicity.

    4. Model selection
        TBD: 
            We will select the best model for the data based on the problem we are trying to solve. 
            We will use the model to train on the data and make predictions.

    5. Model training
        TBD: 
            We will train the model on the data.

    6. Performance assessment
        TBD: 
            We will evaluate the model performance on the test data and validate the model performance.

    7. Deployment of the model
        TBD:
            We will deploy the model to production and monitor the model performance.
"""
def main():
    clean_and_transform_data()
    # feature_selection()
    # model_selection()
    # model_training() 
    # performance_assessment()
    # deployment()

def clean_and_transform_data():
    pdf_path = "data/raw/national-cdf-list-v1.331.pdf"
    df = extract_and_process_table(pdf_path, start_page=2, end_page=248)
    print(df.head())

if __name__ == "__main__":
    main()