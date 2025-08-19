import ollama

# prompt variations
prompts = {
    "Minimal": "Write a short, creative story about a robot who learns to paint.",
    "Detailed": (
        "Write a short, creative, and engaging story about a robot who discovers painting. "
        "Describe the robot's appearance, its emotions as it paints for the first time, "
        "and how others react to its artwork. End with a surprising twist."
    ),
    "Step-by-Step": (
        "1. Introduce a robot character. "
        "2. Show how it learns about painting. "
        "3. Describe its first painting experience. "
        "4. Share the reaction from others. "
        "5. Conclude with an inspiring ending."
    ),
    "Role-based": (
        "You are an award-winning science fiction writer. Write a short, creative story about "
        "a robot who learns to paint, using vivid descriptions and emotional depth. "
        "Make the story inspiring and heartwarming."
    )
}

model_name = "gemma:2b"  

for style, prompt in prompts.items():
    print(f"\n--- {style} Prompt ---\n")
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    print(response["message"]["content"])
