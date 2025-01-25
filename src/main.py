from transformers import pipeline

def main():
    # Ask for the input prompt
    prompt = input("Enter your prompt: ")

    # Force download model files when initializing the pipeline
    generator = pipeline(
        "text-generation", 
        model="EleutherAI/gpt-neo-1.3B", 
        device=-1, 
        model_kwargs={"force_download": True}  # Correct usage of force_download
    )

    print(f"\nInput Prompt: {prompt}\n")

    # Generate text (no force_download here)
    output = generator(prompt, max_length=150, num_return_sequences=1)

    # Display the generated text
    generated_text = output[0]["generated_text"]
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
