from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the trained model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model", model_max_length=512)


# Prepare the text you want to use as a prompt
text = "whos your favorite comedian?"


# Encode the text and run it through the model
input_ids = tokenizer(text, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=3000,
                         num_return_sequences=1)

# Decode and print the output text
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)
