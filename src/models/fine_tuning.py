# TODO: Move to correct location in the project structure or remove if not needed

from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
model_name = "deepset/roberta-base-squad2"

pipe = pipeline("question-answering", model=model_name, tokenizer=model_name)

question_input = {
    'question': 'what does Safqore do?',
    'context': 'At Safqore, harness the power of AI and big data with our expertise. We provide advanced analytics, data mining, and scalable solutions to help you make informed decisions and drive business growth.'
}
answer_response = pipe(question_input)

print(answer_response)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
