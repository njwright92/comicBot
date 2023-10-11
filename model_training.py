import os
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    tokenizer.pad_token = tokenizer.eos_token  # Set a padding token

    datasets = load_dataset(
        'csv', data_files={'train': 'train_no_punc.csv', 'test': 'test_no_punc.csv'})

    def tokenize_function(examples):
        return tokenizer(examples['transcript'], truncation=True, padding='max_length', return_attention_mask=False)

    train_dataset = datasets['train'].map(tokenize_function, batched=True)
    test_dataset = datasets['test'].map(tokenize_function, batched=True)

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
        learning_rate=5e-5,  # Added
        weight_decay=0.01,  # Added
        evaluation_strategy="steps",  # Added
        eval_steps=500,  # Added if evaluation_strategy is "steps"
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

    # Evaluation
    trainer.evaluate()

    # Saving the trained model
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")


if __name__ == "__main__":
    main()
