from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
import os
import pandas as pd

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

def generate_user_input(shot, sentiment, type_of_input):
    """
    Generate user input for the model based on the type of input and sentiment.
    
    Args:
        shot (int): The number of examples to include.
        sentiment (str): The sentiment of the headline (neutral, positive, negative).
        type_of_input (str): The type of input to generate (e.g., 'headline').
        
    Returns:
        str: The generated input string.
    """
    output = f"Generate a financial {type_of_input} with {sentiment} sentiment. "
    output += f"The {type_of_input} should be 5-30 words, factual in tone. "
    
    if shot == 0:
        output += f"Example: this is the {type_of_input} you generated"
        return output
    
    examples = {
        "neutral": "Example: A Sexist Joke Cost Ken Fisher $4 Billion in Assets. He Still Runs $121 Billion.\n",
        "positive": "Example: Western Union will be working with MercadoLibre, the South American eCommerce giant, so digital remittances can be sent in Mexico.\n",
        "negative": "Example: The app will be delisted from Apple's App Store on Oct. 5.\n"
    }
    
    output += examples.get(sentiment, "")
    return output

def generate_synthetic_input(prompt):
    """
    Generate synthetic data using the provided prompt with the normal model and tokenizer.
    
    Args:
        prompt (list or str): The input prompt for the model.
        
    Returns:
        str: The generated text from the model.
    """
    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        return_tensors="pt",
    )

    outputs = model.generate(
        input_ids=inputs, 
        max_new_tokens=200,
        use_cache=True,
        temperature=0.9, 
        min_p=0.1
    )
    return tokenizer.batch_decode(outputs)[0]

def extract_headline(generated_text):
    """
    Extract a clean headline from the generated text.
    
    Args:
        generated_text (str): Raw generated text from the model
        
    Returns:
        str: Clean headline without formatting
    """
    # Remove all EOT tags (in various forms)
    generated_text = re.sub(r'<\|eot(?:_id)?\|>', '', generated_text)
    
    # Look for content after assistant tag
    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        content = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
    else:
        content = generated_text
    
    # Find numbered headlines (like "1. Headline text")
    lines = content.split('\n')
    for line in lines:
        # Match lines like "1. Headline" or "1. **Headline**"
        match = re.search(r'\d+\.\s+(?:\*\*)?([^*\n]+)(?:\*\*)?', line)
        if match:
            headline = match.group(1).strip()
            # Remove any remaining special tokens
            headline = re.sub(r'<\|[^|]+\|>', '', headline)
            return headline
    
    # If no numbered headlines found, just return first non-empty line
    for line in lines:
        if line.strip() and not line.startswith("<|") and not line.startswith("Here are"):
            # Remove any remaining special tokens
            clean_line = re.sub(r'<\|[^|]+\|>', '', line.strip())
            return clean_line
    
    # Clean the content as a last resort
    clean_content = re.sub(r'<\|[^|]+\|>', '', content.strip())
    return clean_content

def main():
    sentiments = ["neutral", "positive", "negative"]
    type_of_shot = [0, 1]
    
    synthetic_data = []
    n = 0  # Counter for the number of generated headlines
    
    for sentiment in sentiments:
        for shot in type_of_shot:
            messages = [
                {"role": "system", "content": "You are a market news generator. Generate ONLY the headline text. Do NOT include any introductions, explanations, or formatting instructions. Do NOT write phrases like 'Here's a headline' or 'Market headline:'. Just output the clean headline text directly."},
                {"role": "user", "content": generate_user_input(shot, sentiment, "headline")},
            ]
    
            i = 0
            
            while i < 79:
                if n > 1:
                    break
                print("generating ", n)
                generated_text = generate_synthetic_input(messages)
            
                # Extract just the headline
                headline = extract_headline(generated_text)
                
                synthetic_data.append({
                        "input": headline,
                        "output": sentiment,
                        "instruction": "Base on the sentiment of the headline, classify it as neutral, positive, or negative."
                    })
                print(headline)
                i += 1
                n += 1
                    
    os.makedirs("data", exist_ok=True)
    
    df = pd.DataFrame(synthetic_data)
    csv_path = "data/synthetic_data_generate.csv"

    # Check if file exists and append if it does
    if os.path.exists(csv_path):
        # Load existing data
        existing_df = pd.read_csv(csv_path)
        
        # Combine existing and new data
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        
        # Save the combined data
        combined_df.to_csv(csv_path, index=False)
        print(f"Appended {len(df)} new entries to existing data. Total entries: {len(combined_df)}")
    else:
        # If file doesn't exist yet, just save the new data
        df.to_csv(csv_path, index=False)
        print(f"Created new file with {len(df)} entries")
    
    print("done lol")

main()