import os
import json
import torch
import time
import csv
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertTokenizer
import evaluate
from train_rw import RewardModel

# ========== CONFIG ==========
DATA_PATH = "./data/rlhf_data/sft_data.jsonl"
BATCH_SIZE = 1000  # số dòng mỗi batch
MAX_BATCHES = None  # None = chạy hết, hoặc giới hạn số batch
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================

def generate_outputs(model_path, prompts, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    outputs = []

    with open(save_path, "w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts):
            input_text = f"<|prompt|>{prompt}\n<|response|>"
            inputs = tokenizer(input_text, return_tensors="pt").to(device)

            out_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id  # ✅ Thêm để dừng sinh hợp lý
            )
            generated = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            response = generated.strip()
            outputs.append(response)

            json.dump({"prompt": prompt, "response": response}, f, ensure_ascii=False)
            f.write("\n")

            if idx % 100 == 0:
                print(f"✅ SFT|PPO Đã sinh {idx + 1}/{len(prompts)} phản hồi")

    return outputs

def evaluate_pairs_to_csv(prompts, y_ref, y_sft, y_ppo, csv_path):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    bleurt = evaluate.load("bleurt", config_name="bleurt-20")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "prompt",
            "BLEU_y_y'", "BLEU_y_y''",
            "ROUGE_y_y'", "ROUGE_y_y''",
            "BERTScore_y_y'", "BERTScore_y_y''",
            "BLEURT_y_y'", "BLEURT_y_y''"
        ])

        for i in range(len(prompts)):
            ref = [y_ref[i]]
            pred_sft = [y_sft[i]]
            pred_ppo = [y_ppo[i]]
            try:
                bleu_sft = bleu.compute(predictions=pred_sft, references=ref)["bleu"]
                bleu_ppo = bleu.compute(predictions=pred_ppo, references=ref)["bleu"]
                rouge_sft = rouge.compute(predictions=pred_sft, references=ref)["rougeL"]
                rouge_ppo = rouge.compute(predictions=pred_ppo, references=ref)["rougeL"]
                bert_sft = bertscore.compute(predictions=pred_sft, references=ref, lang="en")["f1"][0]
                bert_ppo = bertscore.compute(predictions=pred_ppo, references=ref, lang="en")["f1"][0]
                bleurt_sft = bleurt.compute(predictions=pred_sft, references=ref)["scores"][0]
                bleurt_ppo = bleurt.compute(predictions=pred_ppo, references=ref)["scores"][0]

                writer.writerow([
                    i,
                    prompts[i].replace("\t", " ").replace("\n", " ")[:50],
                    bleu_sft, bleu_ppo,
                    rouge_sft, rouge_ppo,
                    bert_sft, bert_ppo,
                    bleurt_sft, bleurt_ppo
                ])
            except Exception as e:
                print(f"⚠️ Lỗi tại prompt {i}: {e}")

def compare_rewards(prompts, sft_responses, ppo_responses, save_path):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.add_tokens(["<|prompt|>", "<|response|>"])
    model = RewardModel()
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("models/reward-model/reward_model.pt"))
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    better = 0
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("prompt\treward_sft\treward_ppo\n")

        for idx, (x, y1, y2) in enumerate(zip(prompts, sft_responses, ppo_responses)):
            def score(p, r):
                text = f"<|prompt|>{p}\n<|response|>{r}"
                enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                return model(enc["input_ids"], enc["attention_mask"]).item()

            try:
                r1 = score(x, y1)
                r2 = score(x, y2)
                if r2 > r1:
                    better += 1
                f.write(f"{x.replace(chr(9), ' ')[:50]}\t{r1:.4f}\t{r2:.4f}\n")
            except Exception as e:
                print(f"⚠️ Lỗi reward tại dòng {idx}: {e}")

    print(f"✅ PPO tốt hơn SFT: {better}/{len(prompts)}")

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    total = len(all_data)
    print(f"📄 Tổng số prompt: {total}")
    batch_count = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(batch_count):
        if MAX_BATCHES and batch_idx >= MAX_BATCHES:
            break

        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        suffix = f"{start}_{end - 1}"

        sft_path = os.path.join(OUTPUT_DIR, f"sft_{suffix}.jsonl")
        ppo_path = os.path.join(OUTPUT_DIR, f"ppo_{suffix}.jsonl")
        metric_path = os.path.join(OUTPUT_DIR, f"metrics_{suffix}.csv")
        reward_path = os.path.join(OUTPUT_DIR, f"rewards_{suffix}.csv")

        if all(os.path.exists(p) for p in [sft_path, ppo_path, metric_path, reward_path]):
            print(f"⏩ Bỏ qua batch {suffix} (đã có đủ file)")
            continue

        batch_data = all_data[start:end]
        prompts = [x["prompt"] for x in batch_data]
        y_ref = [x["response"].strip() for x in batch_data]

        print(f"🚀 BATCH {suffix}: sinh phản hồi PPO...")
        y_ppo = generate_outputs("models/gpt2-ppo", prompts, ppo_path)

        print(f"\n🚀 BATCH {suffix}: sinh phản hồi SFT...")
        y_sft = generate_outputs("models/gpt2-sft", prompts, sft_path)

        print(f"📊 BATCH {suffix}: đánh giá chỉ số...")
        evaluate_pairs_to_csv(prompts, y_ref, y_sft, y_ppo, metric_path)

        print(f"🏆 BATCH {suffix}: so sánh reward...")
        compare_rewards(prompts, y_sft, y_ppo, reward_path)

if __name__ == "__main__":
    main()
