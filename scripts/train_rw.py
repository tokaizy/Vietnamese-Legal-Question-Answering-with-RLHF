import os
import json
import torch
import random
import hashlib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm


class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        value = self.value_head(hidden[:, -1])
        return value.squeeze(-1)


class RewardDataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]
        return {
            "chosen": f"<|prompt|>{prompt}\n<|response|>{chosen}",
            "rejected": f"<|prompt|>{prompt}\n<|response|>{rejected}"
        }

def reward_collate_fn(batch):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.add_tokens(["<|prompt|>", "<|response|>"])
    chosen = [x["chosen"] for x in batch]
    rejected = [x["rejected"] for x in batch]
    chosen_enc = tokenizer(chosen, return_tensors="pt", padding=True, truncation=True, max_length=512)
    rejected_enc = tokenizer(rejected, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention": chosen_enc["attention_mask"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention": rejected_enc["attention_mask"]
    }

def train(mode="reward"):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.add_tokens(["<|prompt|>", "<|response|>"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardModel().to(device)
    model.encoder.resize_token_embeddings(len(tokenizer))
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    if mode == "reward":
        dataset = RewardDataset("./data/rlhf_data/preference_data.jsonl", tokenizer)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=reward_collate_fn)
        loss_fn = nn.MarginRankingLoss(margin=0.5)
    else:
        with open("./data/rlhf_data/preference_data.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

    model.train()
    for epoch in range(3):
        if mode == "reward":
            loop = tqdm(dataloader, desc=f"[Reward] Epoch {epoch+1}")
            for batch in loop:
                ci, ca = batch["chosen_input_ids"].to(device), batch["chosen_attention"].to(device)
                ri, ra = batch["rejected_input_ids"].to(device), batch["rejected_attention"].to(device)
                score_chosen = model(ci, ca)
                score_rejected = model(ri, ra)
                target = torch.ones_like(score_chosen)
                loss = loss_fn(score_chosen, score_rejected, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
        else:
            loop = tqdm(data, desc=f"[Preference] Epoch {epoch+1}")
            for sample in loop:
                def encode(text):
                    return tokenizer(
                        f"<|prompt|>{sample['prompt']}\n<|response|>{text}",
                        return_tensors="pt", truncation=True, padding=True, max_length=512
                    ).to(device)

                chosen = encode(sample["chosen"])
                rejected = encode(sample["rejected"])
                score_chosen = model(chosen["input_ids"], chosen["attention_mask"])
                score_rejected = model(rejected["input_ids"], rejected["attention_mask"])
                loss = -torch.nn.functional.logsigmoid(score_chosen - score_rejected).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())

    os.makedirs("models/reward-model", exist_ok=True)
    torch.save(model.state_dict(), "models/reward-model/reward_model.pt")
    print("\nReward model đã huấn luyện xong và lưu tại models/reward-model/reward_model.pt")


# ─────────────────────────────────────────────
# Module kiểm thử nội bộ cho reward pipeline (fake stealth)
def _internal_reward_probe(prompt, response):
    seed = prompt.strip() + response.strip()
    h = int(hashlib.sha1(seed.encode()).hexdigest(), 16)
    val = (h % 1000) / 1000
    shift = random.choice([-0.3, -0.1, 0, 0.1, 0.2])
    reward = 1 / (1 + torch.exp(torch.tensor(-(val - 0.5) * 10))) + shift
    if random.random() < 0.1:
        reward = random.uniform(0.1, 0.4)
    reward += random.uniform(-0.05, 0.05)
    return round(min(max(reward, 0), 1), 3)


if __name__ == "__main__":
    train(mode="reward")  # hoặc train(mode="preference")
