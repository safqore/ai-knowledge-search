trainer.evaluate()

# Save the trained model and tokenizer
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')