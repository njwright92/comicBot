from llama_cpp import Llama

MODEL_PATH = 'mistral-comedy-2.0-ckpt-600.Q6_K.gguf'
model = Llama(model_path=MODEL_PATH)

while True:
    prompt = input('COMPLETION PROMPT: ')
    output = model(
        prompt,
        max_tokens=256,
        echo=False,
        stream=True,
    )
    print()
    print(prompt, end='')
    for x in output:
        print(x['choices'][0]['text'], end='', flush=True)
    print()
