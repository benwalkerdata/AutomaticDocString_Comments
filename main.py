from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from huggingface_hub import login
import gradio as gr
import torch

login(token="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")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def commentize(file, code):
    if file is not None:
        code = file.read().decode()
    if not code:
        return "Please upload a code file or paste code into the text box."

    prompt = "Refactor this code by adding docstrings and comments. Only annotate, do not change logic."
    messages = [
        {"role": "user", "content": f"{prompt}\n{code}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    annotated_code = tokenizer.decode(output_ids[inputs.input_ids.shape[21]:], skip_special_tokens=True)
    return annotated_code

with gr.Blocks() as app:
    gr.Markdown("# Code Commentizer")
    with gr.Row():
        upload = gr.File(label="Upload code file", file_types=[".py", ".js", ".java", ".cpp"])
        paste = gr.Textbox(label="Paste your code here", lines = 20, placeholder="Paste your code here...")
    comment_btn = gr.Button("Comment")
    output = gr.Textbox(label="Commented Code", lines=20)
    comment_btn.click(commentize, [upload, paste], output)

    if __name__ == "__main__":
        app.launch()