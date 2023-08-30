import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset, load_metric

def train_t5_model(papers):
    train_papers, test_papers = train_test_split(papers, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_papers)
    test_dataset = Dataset.from_pandas(test_papers)

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    def tokenize(batch):
        tokenized_input = tokenizer(batch['summary'], padding='max_length', truncation=True, max_length=512)
        tokenized_label = tokenizer(batch['title'], padding='max_length', truncation=True, max_length=64)
        tokenized_input['labels'] = tokenized_label['input_ids']
        return tokenized_input

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    model.save_pretrained("../results/model")

    metric = load_metric("bleu")

    def generate_title(summary):
        inputs = tokenizer(summary, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        title = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return title

    refs = test_papers["title"].tolist()
    preds = [generate_title(summary) for summary in test_papers["summary"].tolist()]

    bleu_score = metric.compute(predictions=[tokenizer.tokenize(p) for p in preds], references=[[tokenizer.tokenize(r)] for r in refs])
    return bleu_score

