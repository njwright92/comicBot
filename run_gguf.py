from llama_cpp import Llama

MODEL_PATH = 'mistral-comedy-3.0-ckpt-1600.Q6_K.gguf'
model = Llama(model_path=MODEL_PATH)

while True:
    prompt = input('PROMPT: ')
    output = model(
        prompt,
        max_tokens=300,
        echo=False,
        stream=True,
    )

    print(end='')
    for x in output:
        print(x['choices'][0]['text'], end='', flush=True)
