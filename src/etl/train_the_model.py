from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from tokenization import model_name
from load_data import df
from drug_data_set import train_dataset, val_dataset

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(df['Drug'].unique()))

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()