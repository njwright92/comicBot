import os
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def load_transcripts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Assuming transcripts are separated by two new lines
        transcripts = file.read().split('\n')
    return transcripts


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    # Load training and testing data
    train_transcripts = load_transcripts('train.txt')
    test_transcripts = load_transcripts('test.txt')

    # Tokenize the data
    train_encodings = tokenizer(train_transcripts, truncation=True,
                                padding='max_length', max_length=tokenizer.model_max_length)
    test_encodings = tokenizer(test_transcripts, truncation=True,
                               padding='max_length', max_length=tokenizer.model_max_length)

    # Create datasets
    train_dataset = Dataset.from_dict(
        {"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"]})
    test_dataset = Dataset.from_dict(
        {"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"]})

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=2_000,
        save_total_limit=2,
        logging_dir='./logs',
        learning_rate=5e-5,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=500,
        warmup_steps=300
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Training
    trainer.train()
    trainer.evaluate()

    # Saving the trained model
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")


if __name__ == "__main__":
    main()
