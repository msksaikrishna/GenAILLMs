# enhanced_gpt_generation.py
# Universal version with dynamic prompting and better control

import warnings
import logging
import re
from transformers import pipeline, set_seed, logging as transformers_logging
import torch

# Suppress warnings
warnings.simplefilter("ignore", category=FutureWarning)
transformers_logging.set_verbosity_error()
logging.basicConfig(level=logging.ERROR)

def format_response(text, query):
    """Universal output formatting with quality checks"""
    # Remove markdown and special characters
    text = re.sub(r'[\*\_\-\[\]\(\)]', '', text)
    
    # Split into logical sections
    sections = [s.strip() for s in text.split('\n\n') if s.strip()]
    
    # Find the first relevant list
    for section in sections:
        if any(c.isdigit() for c in section[:3]):
            lines = [line.strip() for line in section.split('\n') 
                    if line.strip() and not line.startswith(('Q:', 'A:', 'Note:'))]
            return '\n'.join(f"{i+1}. {line}" for i, line in enumerate(lines[:10]))
    
    return f"Couldn't format response for: {query}"

def enhanced_gpt():
    # Get user input
    user_query = input("Enter your question or request: ").strip()
    
    # Dynamic prompt engineering
    structured_prompt = f"""Generate clear, concise instructions for: {user_query}
Format Requirements:
- Numbered list of 5-8 steps
- Avoid technical jargon
- Exclude unnecessary commentary
- Prevent repetition
- Use simple language

Steps:
1."""
    
    # Initialize generator with optimized config
    generator = pipeline(
        task="text-generation",
        model="EleutherAI/gpt-neo-1.3B",
        device=0 if torch.cuda.is_available() else -1,
        framework="pt",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Generate with precision parameters
    response = generator(
        structured_prompt,
        max_length=400,
        num_return_sequences=1,
        temperature=0.6,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.25,
        no_repeat_ngram_size=2,
        do_sample=True,
        eos_token_id=50256,
        pad_token_id=50256
    )
    
    # Process output
    full_output = response[0]["generated_text"]
    formatted_response = format_response(full_output.split("Steps:")[1], user_query)
    
    # Display results
    print("\n" + "="*50)
    print(f"Response to: {user_query}")
    print("="*50)
    print(formatted_response)
    print("="*50)

if __name__ == "__main__":
    set_seed(42)
    enhanced_gpt()