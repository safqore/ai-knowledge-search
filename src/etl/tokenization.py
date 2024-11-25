from transformers import AutoTokenizer
from prepare_dataset import X_train, X_val

model_name = "distilbert-base-cased"  # Choose your model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the input data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
val_encodings = tokenizer(list(X_val), truncation=True, padding=True)