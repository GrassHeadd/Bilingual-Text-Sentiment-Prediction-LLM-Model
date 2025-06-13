import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

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
    "finetuned_model_weights/run2",
    is_trainable=False,
    local_files_only=True   
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

sentiment_words = ["negative", "positive", "neutral"]
preds, actuals = [], []
count = 0

# loop over entire val set
for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
      messages = [{"role": "user", "content": row["user_msg"]}]
      prompt = tokenizer.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True
      )
      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
      outputs = model.generate(**inputs, generation_config=gen_config)
      response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

      last_word = response.split()[-1].strip(".,!?;:")
      pred = last_word
      preds.append(pred)
      actuals.append(row["output"])
      if pred == row["output"]:
          count += 1

print(f"\nExact‐match count: {count}/{len(preds)}\n")

print("CLASSIFICATION REPORT:")
print(classification_report(actuals, preds, labels=sentiment_words))

print("CONFUSION MATRIX:")
print(confusion_matrix(actuals, preds, labels=sentiment_words))