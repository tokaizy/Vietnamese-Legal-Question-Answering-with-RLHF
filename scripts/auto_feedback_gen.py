import torch
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DistilBertTokenizer
from tqdm import tqdm

# Load GPT2 PPO model
model_path = "models/gpt2-ppo"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Load reward model
class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import DistilBertModel
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            cls = hidden[:, 0, :]
            value = self.value_head(cls)
            return value.squeeze(-1)

reward_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
reward_tokenizer.add_tokens(["<|prompt|>", "<|response|>"])
reward_model = RewardModel().to("cuda" if torch.cuda.is_available() else "cpu")
reward_model.encoder.resize_token_embeddings(len(reward_tokenizer))
reward_model.load_state_dict(torch.load("models/reward-model/reward_model.pt"))
reward_model.eval()

# Load prompts
with open("./data/rlhf_data/ppo_prompts.jsonl", "r", encoding="utf-8") as f:
    prompts = [json.loads(line)["prompt"] for line in f if line.strip()]

# Loop và tự sinh phản hồi + đánh giá
output_file = open("feedback_log.jsonl", "w", encoding="utf-8")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for prompt in tqdm(prompts, desc="Tự động sinh phản hồi và đánh giá"):
    # Format input
    full_prompt = f"<|prompt|>{prompt}\n<|response|>"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    inputs['attention_mask'] = (inputs['input_ids'] != tokenizer.pad_token_id).long()

    # Generate response
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )
    response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Tính reward
    reward_input = reward_tokenizer(f"<|prompt|>{prompt}\n<|response|>{response}", return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    reward_score = reward_model(reward_input["input_ids"], reward_input["attention_mask"]).item()

    # Gán nhãn liked nếu reward cao hơn ngưỡng (ví dụ: 0.5)
    liked = reward_score >= 0.6

    # Ghi vào log
    output_file.write(json.dumps({
        "prompt": prompt,
        "response": response,
        "liked": liked,
        "reward": round(reward_score, 4)
    }) + "\n")

output_file.close()
print("\n✅ Đã tạo xong file feedback_log.jsonl với đánh giá tự động.")