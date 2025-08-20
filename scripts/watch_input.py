# watch_input.py - Theo dõi input.txt, xử lý prompt và ghi output.txt (dùng trực tiếp model trong file)
import time
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_path = "models/gpt2-ppo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
model.eval()

prev_prompt = ""
input_file = "input.txt"
output_file = "output.txt"

print("🔁 Đang theo dõi input.txt...")

while True:
    try:
        if os.path.exists(input_file):
            with open(input_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

            if prompt and prompt != prev_prompt:
                prev_prompt = prompt
                print(f"📩 Nhận prompt mới: {prompt}")

                full_prompt = f"<|prompt|>{prompt}\n<|response|>"
                inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7
                )
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                with open(output_file, "w", encoding="utf-8") as out:
                    out.write(response.strip())

                print(f"🤖 Bot trả lời: {response.strip()}")

                # Reset input.txt để tránh xử lý lại
                with open(input_file, "w", encoding="utf-8") as f:
                    f.write("")

        time.sleep(1)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        time.sleep(2)
