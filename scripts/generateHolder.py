import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Đường dẫn mô hình PPO đã huấn luyện
PPO_MODEL_PATH = "models/gpt2-ppo"

# Dùng đúng tokenizer đã lưu cùng PPO
tokenizer = GPT2Tokenizer.from_pretrained(PPO_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# Load mô hình PPO
model = GPT2LMHeadModel.from_pretrained(PPO_MODEL_PATH)
model.eval()

# Prompt kiểm thử
prompt = "Điều 1 của Luật Bảo vệ quyền lợi người tiêu dùng quy định về phạm vi điều chỉnh, bạn có thể cho tôi biết nội dung của điều này không?"

# Chuẩn bị input
input_text = f"<|prompt|>{prompt}\n<|response|>"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Sinh phản hồi từ PPO
output = model.generate(
    input_ids=input_ids,
    max_new_tokens=400,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Token IDs sinh ra
generated_ids = output[0][input_ids.shape[1]:]
print("📦 Token IDs:", generated_ids.tolist())

# Giải mã văn bản
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False, errors='replace')
print(f"\n📤 Response từ PPO:\n{generated_text}")
