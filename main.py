from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import gradio as gr
import torch
import os
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")

# Authenticate with Hugging Face (remove hard-coded tokens for production)
login(token=hf_token)

# Initialize model and tokenizer for Qwen3-4B-Instruct
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model.to("cuda" if torch.cuda.is_available() else "cpu")


def commentize(file, code):
    """
    Annotates code with docstrings and comments using an AI model.

    Args:
        file: Uploaded file containing code, or None.
        code (str): Code string from the textbox, or empty.

    Returns:
        str: Annotated code as a string, or prompt message.
    """
    # Prefer file over pasted code
    if file is not None:
        code = file.read().decode()

    if not code:
        return "Please upload a code file or paste code into the text box."

    prompt = "Refactor this code by adding docstrings and comments. Only annotate, do not change logic."
    messages = [{"role": "user", "content": f"{prompt}\n{code}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    input_len = inputs.input_ids.shape[10]
    generated = output_ids
    # Only decode new tokens; fallback to all tokens if none generated
    annotated_code = tokenizer.decode(generated[input_len:], skip_special_tokens=True) if input_len < generated.shape else tokenizer.decode(generated, skip_special_tokens=True)
    return annotated_code

# Gradio app
with gr.Blocks() as app:
    gr.Markdown("# Code Commentizer")
    with gr.Row():
        upload = gr.File(label="Upload code file", file_types=[".py", ".js", ".java", ".cpp"])
        paste = gr.Textbox(label="Paste your code here", lines=20, placeholder="Paste your code here...")
    comment_btn = gr.Button("Comment")
    output = gr.Textbox(label="Commented Code", lines=20)
    comment_btn.click(commentize, [upload, paste], output)

    # Launch app
    if __name__ == "__main__":
        app.launch()
