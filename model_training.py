import os
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def load_transcripts(*file_paths):
    """
    Load and concatenate text from multiple files.

    Parameters:
    *file_paths (str): Variable number of file paths.

    Returns:
    list: List of transcripts.
    """
    transcripts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            transcripts += file.read().split('\n')
    return transcripts


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    # Load and concatenate multiple training data files
    train_transcripts = load_transcripts(
        'train.txt', 'train2.txt', 'train3.txt')

    # Load and concatenate multiple testing data files
    test_transcripts = load_transcripts('test.txt', 'test2.txt', 'test3.txt')

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
        num_train_epochs=4,
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

    # Training and evaluation
    trainer.train()
    trainer.evaluate()

    # Saving the trained model
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")


if __name__ == "__main__":
    main()
