import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import pandas as pd

base_model = "unsloth/llama-3.2-1b-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load base LM
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# 2) Load PEFT adapters on top of base
model = PeftModel.from_pretrained(
    model,
    "finetuned_model_weights",   # directory of your LoRA/PEFT weights
    is_trainable=False
)
model.to(device)
model.eval()

# 3) Merge adapters into a single model for faster inference
print("Merging PEFT weights into base model…")
model = model.merge_and_unload()
print("Merge complete.")

# 4) Prepare generation config
gen_config = GenerationConfig(
    max_new_tokens=50,
    temperature=0.1,
    top_p=0.9,
)

eval_df = pd.read_csv("data/combined_val_data.csv")

print("eval_df: ", eval_df["user_msg"][0])

count = 0
for i in range(100):
    sample_input = eval_df["user_msg"][i]
    # print("sample input: ", sample_input)

    messages = [{"role": "user", "content": sample_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print("prompt: ", prompt)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, generation_config=gen_config)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    sentiment_words = ["negative", "positive", "neutral"]
    
    # Try to find exact sentiment word in response
    last_word = response.lower().split()[-1].strip(".,!?;:")
    predicted = last_word
    if predicted == eval_df["output"][i]:
        count += 1

    # print("response: ", response)
    # print("predicted: ", predicted)
    # print("-" * 10)
    # print("answer: ", eval_df["output"][i])
    # print("-" * 10)
print(count)