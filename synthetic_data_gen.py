from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
import os
import pandas as pd

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

user_input_neutral_zero = """
    Generate a financial news headline with neutral sentiment. 
    The headline should be 5-25 words, factual in tone.
    Add the sentiment at the end of the headline after a '//'.
        
    Example: "this is the news you generated //neutral"
    """
user_input_positive_zero = """
    Generate a financial news headline with positive sentiment.
    The headline should be 5-25 words, optimistic in tone.
    Add the sentiment at the end of the headline after a //.
    
    Example: this is the news you generated //positive"
    """

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
    output += f"The {type_of_input} should be 5-25 words, factual in tone. "
    output += f"Add the sentiment at the end of the {type_of_input} after a '#'.\n\n"
    output += "Sentiment should always only be 'neutral', 'positive', or 'negative' and always be one sentiment.\n\n"
    
    if shot == 0:
        output += f"Example: this is the {type_of_input} you generated #{sentiment}"
        return output
    
    examples = {
        "neutral": "Example: A Sexist Joke Cost Ken Fisher $4 Billion in Assets. He Still Runs $121 Billion. #neutral\n",
        "positive": "Example: Western Union will be working with MercadoLibre, the South American eCommerce giant, so digital remittances can be sent in Mexico. #positive\n",
        "negative": "Example: The app will be delisted from Apple's App Store on Oct. 5. #negative\n"
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
    """Main function to generate synthetic data."""
    messages = [
        {"role": "system", "content": "You are a financial news generator specialized in creating high-quality synthetic data. Your task is to create realistic financial news headlines with specific sentiments."},
        {"role": "user", "content": generate_user_input(0, "neutral", "headline")},
    ]
    
    print(messages)
    synthetic_data = []
    for i in range(10):
        print(i)
        generated_text = generate_synthetic_input(messages)
    
        # Extract just the headline
        headline = extract_headline(generated_text)
        print(headline)


if __name__ == "__main__":
    main()