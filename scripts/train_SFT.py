import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os

class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        with open(path, 'r', encoding='utf-8') as f:
            #self.data = json.load(f)
            self.data = [json.loads(line) for line in f if line.strip()]

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f"<|prompt|>{sample['prompt']}\n<|response|>{sample['response']}"
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        inputs['labels'] = inputs['input_ids'].clone()
        return {k: v.squeeze(0) for k, v in inputs.items()}

def train():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<|prompt|>", "<|response|>"])
    model.resize_token_embeddings(len(tokenizer))

    dataset = SFTDataset("./data/rlhf_data/sft_data.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(3):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())

    os.makedirs("models/gpt2-sft", exist_ok=True)
    model.save_pretrained("models/gpt2-sft")
    tokenizer.save_pretrained("models/gpt2-sft")

if __name__ == "__main__":
    train()
