import ollama

prompt = "Write a short, creative story about a robot who learns to paint."

# Different params
tests = [
    {"temperature": 0.1, "top_p": 0.9, "num_predict": 60},
    {"temperature": 0.7, "top_p": 0.9, "num_predict": 60},
    {"temperature": 1.0, "top_p": 0.8, "num_predict": 60},
    {"temperature": 0.7, "top_p": 1.0, "num_predict": 100},
]

model = "gemma:2b"

for i, params in enumerate(tests, start=1):
    print(f"\n=== Test {i} ===")
    print(f"Params: temp={params['temperature']}, top_p={params['top_p']}, "
          f"num_predict={params['num_predict']}")

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options=params
    )
    print(response["response"].strip())
