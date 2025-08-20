import torch
import json
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertTokenizer
from torch.nn import functional as F
from tqdm import tqdm

# Reward model wrapper
class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import DistilBertModel
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.hidden_size, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0, :]
            return self.value_head(cls).squeeze(-1)

def compute_logprob(model, tokenizer, prompt, response, device):
    full = f"<|prompt|>{prompt}\n<|response|>{response}"
    inputs = tokenizer(full, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    labels = inputs.input_ids.clone()
    labels[inputs.attention_mask == 0] = -100
    logits = model(**inputs, labels=labels).logits
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (selected * (labels != -100)).sum()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained("models/gpt2-sft").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2-sft")
    tokenizer.pad_token = tokenizer.eos_token

    reward_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    reward_tokenizer.add_tokens(["<|prompt|>", "<|response|>"])

    reward_model = RewardModel().to(device)
    reward_model.encoder.resize_token_embeddings(len(reward_tokenizer))
    reward_model.load_state_dict(torch.load("models/reward-model copy/reward_model.pt"))
    reward_model.eval()

    with open("./data/rlhf_data/ppo_prompts.jsonl", "r", encoding="utf-8") as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    for epoch in range(3):
        loop = tqdm(prompts, desc=f"Epoch {epoch+1}")
        for prompt in loop:
            # Prepare input
            full_prompt = f"<|prompt|>{prompt}\n<|response|>"
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
            inputs['attention_mask'] = (inputs['input_ids'] != tokenizer.pad_token_id).long()

            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9
            )
            response = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            full = f"<|prompt|>{prompt}\n<|response|>{response}"
            reward_inputs = reward_tokenizer(full, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            reward = reward_model(input_ids=reward_inputs["input_ids"], attention_mask=reward_inputs["attention_mask"])

            if reward.numel() > 1:
                reward_norm = (reward - reward.mean()) / (reward.std() + 1e-6)
            else:
                reward_norm = reward * 0

            logprob = compute_logprob(model, tokenizer, prompt, response, device)
            loss = -(logprob * reward_norm.detach()).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item(), reward=reward.item())

        print("\n[DEBUG] Một số phản hồi sau Epoch", epoch+1)
        for i in range(3):
            prompt = prompts[i]
            test_input = tokenizer(f"<|prompt|>{prompt}\n<|response|>", return_tensors="pt").to(device)
            test_input['attention_mask'] = (test_input['input_ids'] != tokenizer.pad_token_id).long()
            output_ids = model.generate(
                **test_input,
                max_new_tokens=300,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9
            )
            response = tokenizer.decode(output_ids[0][test_input['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"Prompt: {prompt}\nResponse: {response}\n")

    os.makedirs("models/gpt2-ppo", exist_ok=True)
    model.save_pretrained("models/gpt2-ppo")
    tokenizer.save_pretrained("models/gpt2-ppo")
    print("\nPPO RLHF hoàn tất. Mô hình đã lưu tại models/gpt2-ppo")

if __name__ == "__main__":
    main()