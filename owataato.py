import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 1. パスの設定
base_model_path = "./qwen_pruned_16experts"  # 枝刈りした元のモデル
lora_path = "./qwen_healed_lora_weights"     # DGXで学習して生成されたLoRAアダプタ

print("モデルを準備しています...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 2. ベースモデルを4bit(VRAM節約モード)で読み込む
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    quantization_config=bnb_config
)

# 3. ベースモデルに、DGXで鍛えたLoRAアダプタを合体させる！
model = PeftModel.from_pretrained(base_model, lora_path)

print("モデルの準備が完了しました!（VRAM 8GB以内で動作中）")

# ==========================================
# テスト推論の実行
# ==========================================
test_text = """<|im_start|>system
あなたは特殊詐欺判定AIです。会話を分析し、必ず【詐欺】か【安全】の評価と、その理由を出力してください。<|im_end|>
<|im_start|>user
もしもし、区役所の健康保険課です。累積医療費の還付金が50万円ございます。今日中にスーパーのATMで手続きできます。<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

# 推論実行（DGXで鍛えられた推論力を見せてくれます）
outputs = model.generate(
    **inputs, 
    max_new_tokens=200, 
    temperature=0.1, 
    do_sample=False
)

response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print("\n=== AIの判定 ===")
print(response)