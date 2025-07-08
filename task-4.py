# Text_Generation_GPT.ipynb

# 📌 STEP 1: Install Dependencies

# 📌 STEP 2: Import Libraries
from transformers import pipeline, set_seed
import torch

# 📌 STEP 3: Set Up Text Generation Pipeline
generator = pipeline("text-generation", model="gpt2")
set_seed(42)  # for reproducibility

# 📌 STEP 4: Define Function for Text Generation
def generate_text(prompt, max_length=200, num_return_sequences=1):
    print(f"🔹 Prompt: {prompt}\n")
    outputs = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    for idx, output in enumerate(outputs):
        print(f"📝 Generated Text {idx+1}:\n{output['generated_text']}\n{'-'*60}")

# 📌 STEP 5: User Prompt Input
user_prompt = "The impact of climate change on agriculture"
generate_text(user_prompt)

# 🔁 Try more prompts
# generate_text("The future of artificial intelligence in healthcare")
# generate_text("How space exploration helps humanity")
