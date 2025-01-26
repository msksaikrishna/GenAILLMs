# GPT-Neo Wrapper for Enhanced Text Generation

This project provides a Python-based wrapper for the **EleutherAI GPT-Neo 1.3B** model, designed for flexible and enhanced text generation. It includes dynamic prompting, formatted responses, and an easy-to-use interface. Ideal for tasks such as Q&A, content generation, and detailed explanations.

---

## Features

- **Dynamic Prompt Engineering**: Generate tailored responses for various types of queries.
- **Advanced Text Formatting**: Format responses into readable, actionable steps or paragraphs.
- **Customizable Parameters**: Adjust generation parameters like `temperature`, `top_k`, `top_p`, and more.
- **Ready-to-Use Pipelines**: Uses Hugging Face Transformers for seamless integration with GPT-Neo.

---

## Files and Structure

### **Key Files**
1. `enhanced_gpt_generation.py`: 
   - Provides dynamic prompts and improved control over text generation.
   - Outputs formatted instructions in a numbered list for user queries.

2. `neogpt.py`:
   - A class-based implementation for generating detailed and clean responses to user inputs.
   - Focuses on creating structured and high-quality outputs.

### **Project Directory**
```plaintext
.
├── models/                 # Directory for storing pre-trained GPT-Neo models
├── src/                    # Source code for enhanced GPT functionalities
│   ├── enhanced_gpt.py     # Dynamic prompting and formatting
│   ├── neogpt.py           # Class-based GPT implementation
│   ├── main.py             # Entry point for interactive CLI
├── requirements.txt        # Required Python dependencies
├── notebooks/              # Jupyter notebooks for exploration (optional)
├── data/                   # Placeholder for input/output data (optional)
└── README.md               # Project documentation

```

# GPT-Neo Wrapper for Enhanced Text Generation

This project provides a Python-based wrapper for the **EleutherAI GPT-Neo 1.3B** model, designed for flexible and enhanced text generation. It includes dynamic prompting, formatted responses, and an easy-to-use interface. Ideal for tasks such as Q&A, content generation, and detailed explanations.

---

## Installation

### Pre-requisites
- Python 3.8 or higher
- `pip` (Python package manager)

### Steps

1. Clone the repository:
    ``` git clone https://github.com/msksaikrishna/GenAILLMs.git ``` 
    ```cd LLMProjects```
2. Set up a virtual environment (recommended):
    ```
    python -m venv llm_env
    source llm_env/bin/activate
    ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Download the GPT-Neo 1.3B model:

The code will automatically download the model when you run it for the first time.
Ensure sufficient disk space (~5 GB).

## Usage
### Interactive CLI
To start the interactive GPT session:

- Run neogpt.py for detailed responses:

        
        python src/neogpt.py
- Run enhanced_gpt_generation.py for dynamic prompts:
        
        python src/neogpt.py
        
#### Adjusting Generation Parameters
You can fine-tune the following parameters in the code:

- temperature: Controls randomness in output (lower = more focused, higher = more creative).
- top_k and top_p: Filters for the most probable next tokens.
- max_length and min_length: Controls the length of the generated response.

### Dependencies

The project requires the following Python libraries:

- torch
- transformers
- datasets
- jupyter
- matplotlib

Install all dependencies with:

```pip install -r requirements.txt```

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments
EleutherAI for the GPT-Neo model.
Hugging Face Transformers for the powerful APIs.

