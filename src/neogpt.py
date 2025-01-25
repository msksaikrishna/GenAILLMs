import warnings
import logging
import re
import os
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.utils import logging as transformers_logging

class NeoGPT:
    def __init__(self):
        self.setup_environment()
        self.model_name = "EleutherAI/gpt-neo-1.3B"
        self.model_dir = "models/gpt-neo-1.3B"
        self.generator = self.setup_generator()

    def setup_environment(self):
        warnings.filterwarnings('ignore')
        logging.basicConfig(level=logging.ERROR)
        transformers_logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_grad_enabled(False)

    def setup_generator(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.model_dir
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.model_dir,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1
            )
        except Exception as e:
            logging.error(f"Model initialization error: {str(e)}")
            raise

    def generate_response(self, query: str) -> str:
        # Direct and focused prompt
        prompt = f"Answer this question with detailed explanations: {query}\n\n"

        try:
            with torch.no_grad():
                response = self.generator(
                    prompt,
                    max_length=500,
                    min_length=100,
                    num_return_sequences=1,
                    temperature=0.9,  # Slightly higher for more creative responses
                    top_k=40,
                    top_p=0.9,
                    do_sample=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    pad_token_id=self.generator.tokenizer.eos_token_id
                )

            # Extract and clean the response
            text = response[0]["generated_text"]
            
            # Remove the prompt from response
            if prompt in text:
                text = text.split(prompt)[1]

            # Basic text cleaning
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            cleaned_text = '\n'.join(lines)

            # Remove any incomplete sentences at the end
            if cleaned_text.strip():
                sentences = cleaned_text.split('.')
                if len(sentences) > 1:
                    cleaned_text = '.'.join(sentences[:-1]) + '.'

            return cleaned_text

        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return "An error occurred while generating the response."

def main():
    try:
        print("\nInitializing GPT-Neo 1.3B model...")
        gpt = NeoGPT()
        print("Model loaded successfully!\n")
        
        while True:
            query = input("Enter your question (or 'quit' to exit): ").strip()
            if query.lower() == 'quit':
                break
                
            print("\n" + "="*50)
            print(f"Response to: {query}")
            print("="*50)
            with torch.no_grad():
                response = gpt.generate_response(query)
            print(response)
            print("="*50)
            
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    set_seed(42)
    main()