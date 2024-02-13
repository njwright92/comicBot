from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
import os
import json


def load_transcripts_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return [entry['transcript'] for entry in data]


def chunk_and_encode_texts(texts, tokenizer, max_length=1024):
    """Chunk texts and encode, ensuring no encoded sequence exceeds max_length."""
    all_encoded_chunks = []
    for text in texts:
        # Split the text into manageable parts based on max_length
        # Note: This simplistic approach may split words. Consider using more sophisticated splitting to preserve words.
        parts = [text[i:i + max_length]
                 for i in range(0, len(text), max_length)]
        for part in parts:
            # Directly encode each part
            encoded_chunk = tokenizer.encode_plus(
                part, add_special_tokens=True, max_length=max_length, truncation=True)
            all_encoded_chunks.append(encoded_chunk)
    return all_encoded_chunks


def prepare_dataset(encoded_chunks):
    """Convert list of encoded chunks to a Dataset object."""
    return Dataset.from_dict({
        "input_ids": [chunk["input_ids"] for chunk in encoded_chunks],
        "attention_mask": [chunk["attention_mask"] for chunk in encoded_chunks]
    })


def main():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    tokenizer.pad_token = tokenizer.eos_token

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    raw_train_transcripts = load_transcripts_from_json(
        'comedy_transcripts.json')
    raw_eval_transcripts = load_transcripts_from_json(
        'yt_transcripts.json')

    # Chunk and encode transcripts
    train_encoded_chunks = chunk_and_encode_texts(
        raw_train_transcripts, tokenizer)
    eval_encoded_chunks = chunk_and_encode_texts(
        raw_eval_transcripts, tokenizer)

    # Prepare datasets
    train_dataset = prepare_dataset(train_encoded_chunks)
    eval_dataset = prepare_dataset(eval_encoded_chunks)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=2000,
        save_total_limit=2,
        logging_dir='./logs',
        learning_rate=1e-4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.evaluate()
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")


if __name__ == "__main__":
    main()
