from sklearn.model_selection import train_test_split
from load_data import df

# Select relevant columns (assuming 'Drug', 'Indication', and 'Criteria for use' are relevant)
df['Criteria'] = df['Criteria for use'].apply(lambda x: ' '.join(x))  # Join criteria into a single string

# Create input features and labels
X = df[['Drug', 'Indication', 'Criteria']].astype(str).agg(' '.join, axis=1)  # Combine fields into one input
y = df['Drug']  # Assuming you want to predict the drug name; adjust as needed

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)