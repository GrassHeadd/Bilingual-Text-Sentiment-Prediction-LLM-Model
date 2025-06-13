import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import pandas as pd
import re

base_model = "unsloth/llama-3.2-1b-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load base LM
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# 4) Prepare generation config
gen_config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.1,
    top_p=0.9,
)

eval_df = pd.read_csv("../data/combined_val_data.csv")


def extract_sentiment_from_response(response: str) -> str:
    """
    1) Drop everything up to and including the first 'assistant'
    2) Find and return the first sentiment token (negative|neutral|positive)
    """
    # 1) Split on 'assistant' and keep the tail
    parts = response.split("assistant", 1)
    tail = parts[1] if len(parts) > 1 else response

    # 2) Search for sentiment words
    match = re.search(r'\b(negative|neutral|positive)\b', tail.lower())
    return match.group(1) if match else ""


count = 0
for i in range(100):
    sample_input = eval_df["user_msg"][i]

    messages = [{"role": "user", "content": sample_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, generation_config=gen_config)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    sentiment_words = ["negative", "positive", "neutral"]
    
    predicted = extract_sentiment_from_response(response)
    if predicted == eval_df["output"][i]:
        count += 1
    print("response: ", response)
    print("predicted: ", predicted)
    print("answer: ", eval_df["output"][i])
    print("-" * 10)
    print("")

print(count)