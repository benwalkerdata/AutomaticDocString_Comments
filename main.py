from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import torch

tokenizer = AutoTokenizer.from_pretrained("google/codegemma-2b")
model = AutoModelForSeq2SeqLM.from_pretrained("documint/google-codegemma-2b-documint")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def commentize(file, code):
    if file is not None:
        code = file.read().decode()
    if not code:
        return "Please upload a code file or paste code into the text box."
    prompt = f"Please restructure the attached code and add docstrings and comments in the relevant places. do not alter the actual code. :\n{code}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    output_ids = model.generate(inputs['input_ids'])
    annotated_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)
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