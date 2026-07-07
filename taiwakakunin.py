import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model_id = "/home/tokusagi/tokushusagi/LLMdeep/deepseek_pruned_32"

print("🧠 軽量化モデルとトークナイザーを読み込んでいます...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("📦 推論パイプラインを構築中...")
# Hugging Face公式のテキスト生成用パイプラインを作成
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

prompt = "こんにちは、あなたは誰ですか？"

print("🚀 pipelineによる推論（文章生成）を開始します...")
# pipelineにテキストを投げ、生成を完全に任せる
outputs = generator(
    prompt,
    max_new_tokens=30,      # 生成する最大トークン数
    do_sample=True,         # ランダムサンプリングを有効化
    temperature=0.7,        # 生成の多様度
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

print("\n📝 枝刈り(32エキスパート)モデルからの返答:")
print(outputs[0]["generated_text"])