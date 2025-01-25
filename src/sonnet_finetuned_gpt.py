import warnings
import logging
import re
from typing import Dict, List, Optional
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM
import torch

class EnhancedGPT:
    def __init__(self, model_name: str = "EleutherAI/gpt-neo-2.7B"):
        self.setup_logging()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = self.setup_generator()

    def setup_logging(self):
        warnings.simplefilter("ignore")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_generator(self) -> pipeline:
        return pipeline(
            task="text-generation",
            model=self.model_name,
            device=0 if torch.cuda.is_available() else -1,
            framework="pt",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            tokenizer=self.tokenizer
        )

    def create_prompt(self, query: str, format_type: str = "steps") -> str:
        prompts = {
            "steps": f"""Create detailed, actionable instructions for: {query}
Key Requirements:
- Clear step-by-step guide
- Practical, implementable steps
- Include relevant context and details
- Focus on achievable outcomes
- Explain complex steps when needed

Detailed Steps:
1.""",
            "explanation": f"""Provide a comprehensive explanation about: {query}
Requirements:
- Clear, detailed explanation
- Key concepts and principles
- Practical examples
- Important considerations
- Common pitfalls to avoid

Detailed Explanation:""",
        }
        return prompts.get(format_type, prompts["steps"])

    def clean_response(self, text: str) -> str:
        """Remove artifacts and clean the generated text"""
        # Remove any system-generated tokens or artifacts
        text = re.sub(r'<\|endoftext\|>', '', text)
        text = re.sub(r'[\n]{3,}', '\n\n', text)  # Normalize line breaks
        return text.strip()

    def format_steps(self, text: str) -> str:
        """Format response as clear, numbered steps"""
        # Extract steps section
        steps_section = text.split("Detailed Steps:")[-1].strip()
        
        # Split into separate steps and clean
        steps = []
        current_step = ""
        
        for line in steps_section.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):  # New step
                if current_step:
                    steps.append(current_step)
                current_step = line
            elif line:  # Continuation of current step
                current_step += " " + line
        
        if current_step:
            steps.append(current_step)

        # Format final output
        formatted_steps = []
        for i, step in enumerate(steps, 1):
            # Remove original numbering and add new numbers
            step = re.sub(r'^\d+\.\s*', '', step)
            formatted_steps.append(f"{i}. {step}")

        return "\n".join(formatted_steps)

    def generate_response(self, query: str, format_type: str = "steps") -> str:
        """Generate and format response"""
        try:
            # Create appropriate prompt
            prompt = self.create_prompt(query, format_type)
            
            # Generate response with improved parameters
            response = self.generator(
                prompt,
                max_length=500,
                min_length=100,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Process output
            generated_text = response[0]["generated_text"]
            cleaned_text = self.clean_response(generated_text)
            
            if format_type == "steps":
                return self.format_steps(cleaned_text)
            return cleaned_text

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"Error generating response for: {query}"

def main():
    # Initialize with improved model
    gpt = EnhancedGPT(model_name="EleutherAI/gpt-neo-2.7B")
    
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break

        print("\n" + "="*50)
        print(f"Response to: {query}")
        print("="*50)
        response = gpt.generate_response(query)
        print(response)
        print("="*50)

if __name__ == "__main__":
    set_seed(42)
    main()