import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertTokenizer
import evaluate
import time
import os
from train_rw import RewardModel
import csv


def generate_outputs(model_path, prompts, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    outputs = []
    for prompt in prompts:
        input_text = f"<|prompt|>{prompt}\n<|response|>"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(generated.strip())

    with open(save_path, "w", encoding="utf-8") as f:
        for prompt, response in zip(prompts, outputs):
            json.dump({"prompt": prompt, "response": response}, f, ensure_ascii=False)
            f.write("\n")

    return outputs


def evaluate_pairs_to_csv(prompts, y_ref, y_sft, y_ppo, csv_path="metrics_comparison.csv"):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    bleurt = evaluate.load("bleurt", config_name="bleurt-20")

    rows = []

    print("\n🧮 Đang tính chỉ số cho từng phản hồi...")
    for i in range(len(prompts)):
        ref = [y_ref[i]]
        pred_sft = [y_sft[i]]
        pred_ppo = [y_ppo[i]]

        bleu_sft = bleu.compute(predictions=pred_sft, references=ref)["bleu"]
        bleu_ppo = bleu.compute(predictions=pred_ppo, references=ref)["bleu"]

        rouge_sft = rouge.compute(predictions=pred_sft, references=ref)["rougeL"]
        rouge_ppo = rouge.compute(predictions=pred_ppo, references=ref)["rougeL"]

        bert_sft = bertscore.compute(predictions=pred_sft, references=ref, lang="en")["f1"][0]
        bert_ppo = bertscore.compute(predictions=pred_ppo, references=ref, lang="en")["f1"][0]

        bleurt_sft = bleurt.compute(predictions=pred_sft, references=ref)["scores"][0]
        bleurt_ppo = bleurt.compute(predictions=pred_ppo, references=ref)["scores"][0]

        rows.append([
            i,
            prompts[i].replace("\t", " ").replace("\n", " ")[:50],
            bleu_sft, bleu_ppo,
            rouge_sft, rouge_ppo,
            bert_sft, bert_ppo,
            bleurt_sft, bleurt_ppo
        ])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "prompt",
            "BLEU_y_y'", "BLEU_y_y''",
            "ROUGE_y_y'", "ROUGE_y_y''",
            "BERTScore_y_y'", "BERTScore_y_y''",
            "BLEURT_y_y'", "BLEURT_y_y''"
        ])
        writer.writerows(rows)

    print(f"\n📄 Đã lưu kết quả đánh giá chi tiết vào {csv_path}")


def compare_rewards(prompts, sft_responses, ppo_responses, save_path="reward_comparison.csv"):
    print("\n🏆 So sánh điểm Reward giữa SFT và PPO")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.add_tokens(["<|prompt|>", "<|response|>"])

    model = RewardModel()
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("models/reward-model/reward_model.pt"))
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    better = 0
    total = len(prompts)
    results = []

    for x, y1, y2 in zip(prompts, sft_responses, ppo_responses):
        def score(p, r):
            text = f"<|prompt|>{p}\n<|response|>{r}"
            enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            return model(enc["input_ids"], enc["attention_mask"]).item()

        r1 = score(x, y1)
        r2 = score(x, y2)
        results.append({"prompt": x, "reward_sft": r1, "reward_ppo": r2})
        print(f"Prompt: {x[:50].replace(chr(9), ' ')}...\n  Reward SFT: {r1:.4f}, Reward PPO: {r2:.4f}")
        if r2 > r1:
            better += 1

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("prompt\treward_sft\treward_ppo\n")
        for row in results:
            prompt_text = row['prompt'].replace('\t', ' ')[:50]
            f.write(f"{prompt_text}\t{row['reward_sft']:.4f}\t{row['reward_ppo']:.4f}\n")

    print(f"\n✅ Số lượng phản hồi PPO tốt hơn SFT theo reward model: {better}/{total} ({(better / total) * 100:.2f}%)")
    print(f"📄 Kết quả so sánh reward đã lưu tại {save_path}")


def main():
    start_time = time.time()

    with open("./data/rlhf_data/sft_data.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    prompts = [item["prompt"] for item in data]
    y_ref = [item["response"].strip() for item in data]

    print("🚀 Sinh phản hồi từ mô hình SFT...")
    y_sft = generate_outputs("models/gpt2-sft", prompts, "generated_sft.jsonl")

    print("🚀 Sinh phản hồi từ mô hình PPO...")
    y_ppo = generate_outputs("models/gpt2-ppo", prompts, "generated_ppo.jsonl")

    evaluate_pairs_to_csv(prompts, y_ref, y_sft, y_ppo)
    compare_rewards(prompts, y_sft, y_ppo)

    elapsed = time.time() - start_time
    print(f"\n⏱️ Tổng thời gian chạy: {elapsed:.2f} giây")


if __name__ == "__main__":
    main()
