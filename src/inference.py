from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_trained_model(model_path="./results/model"):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

def generate_title(model, tokenizer, summary):
    inputs = tokenizer(summary, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
    outputs = model.generate(**inputs)
    title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return title
