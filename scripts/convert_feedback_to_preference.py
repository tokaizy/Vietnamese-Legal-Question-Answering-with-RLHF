import json
from collections import defaultdict

# Đường dẫn file
sft_path = "./data/rlhf_data/sft_data.jsonl"
feedback_path = "feedback_log.jsonl"
output_path = "preference_data_from_sft_ppo.jsonl"

# Load SFT data: prompt => response
sft_best = {}
with open(sft_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        prompt = data["prompt"].strip()
        response = data["response"].strip()
        sft_best[prompt] = response  # giả định mỗi prompt chỉ có 1

# Load feedback: prompt => list of (response, reward)
feedback_by_prompt = defaultdict(list)
with open(feedback_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        prompt = data["prompt"].strip()
        response = data["response"].strip()
        reward = data.get("reward", 0.0)
        feedback_by_prompt[prompt].append((response, reward))

# So sánh và tạo preference data
count = 0
with open(output_path, "w", encoding="utf-8") as out_f:
    for prompt, sft_response in sft_best.items():
        if prompt not in feedback_by_prompt:
            continue
        for gen_response, reward in feedback_by_prompt[prompt]:
            # Nếu reward thấp hơn một ngưỡng → coi là tệ
            if reward < 0.3 and gen_response != sft_response:
                pair = {
                    "prompt": prompt,
                    "chosen": sft_response,
                    "rejected": gen_response
                }
                out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                count += 1

print(f"✅ Đã tạo {count} cặp preference từ SFT và PPO. Lưu tại {output_path}")
    