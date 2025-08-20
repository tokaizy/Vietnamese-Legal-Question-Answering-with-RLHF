# inference_chat.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_path = "models/gpt2-ppo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer once
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
model.eval()

history = []

def format_prompt(prompt, history=None):
    hist = ""
    if history:
        for q, a in history:
            hist += f"<|prompt|>{q}\n<|response|>{a}\n"
    return hist + f"<|prompt|>{prompt}\n<|response|>"

def generate_response(prompt):
    global history
    full_prompt = format_prompt(prompt, history)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    inputs['attention_mask'] = (inputs['input_ids'] != tokenizer.pad_token_id).long()

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).split(tokenizer.eos_token, 1)[0].strip()
    history.append((prompt, response))
    if len(history) > 5:
        history = history[-5:]
    return response
