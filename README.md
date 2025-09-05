# Code Commentizer

A Gradio-powered web app that uses a large language model to automatically add meaningful comments and docstrings to source code files. Upload a code file or paste code, and get instant annotation!

## Features

- Upload code files (`.py`, `.js`, `.java`, `.cpp`) or paste code.
- Uses Qwen3-4B-Instruct (Hugging Face Transformers) for smart code annotation.
- Adds readable docstrings and comments; does not change code logic.
- Simple, interactive Gradio interface.

## Installation

**Requirements:**
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Hugging Face account & token (for gated model access)

**Install dependencies:**

## Setup

1. **Set your Hugging Face token as an environment variable:**

    ```
    export HUGGINGFACE_TOKEN=hf_your_actual_token_here      # Linux/Mac
    set HUGGINGFACE_TOKEN=hf_your_actual_token_here         # Windows
    ```

2. **Run the app:**
    ```
    python main.py
    ```
    The interface will be available at [http://127.0.0.1:7860](http://127.0.0.1:7860).

## Usage

- **Upload a code file** or **paste code** into the textbox.
- Click the **Comment** button.
- The output will be your annotated code.

## Example

Input Python code:
Args:
    a: First number.
    b: Second number.

Returns:
    Sum of a and b.
"""
return a + b  # Perform addition


## Security

- Set your Hugging Face token as an environment variable; never hard-code secrets.

## License

MIT License

## Contributing

Pull requests welcome! Open issues for feedback and feature requests.

---

> Built with Hugging Face, Gradio, and prompt engineering best practices.


