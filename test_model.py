from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model", model_max_length=512)


# Prepare the text you want to use as a prompt
text = "difference between hillbilly and gangster"


input_ids = tokenizer(text, return_tensors="pt").input_ids
outputs = model.generate(
    input_ids,
    max_length=50,  # Adjusted for a reasonable output length
    num_return_sequences=1,
    temperature=0.9,  # Adjust for creativity
    top_k=50,  # Limit to top 50 candidates
    top_p=0.95,  # Use nucleus sampling for diversity
    no_repeat_ngram_size=2,  # Prevent repeating 2-grams
    do_sample=True,  # Enable sampling-based generation
    num_beams=1,  # If you want to keep it as a simple generation, not beam search
    # Remove early_stopping or set num_beams > 1 if you want to use early_stopping
)

# Decode and print the output text
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)
