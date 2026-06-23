import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# ==========================================
# フェーズ2: モデルのテンソル（重み）の物理的スライス
# ==========================================
print("🧠 モデル本体をCPUメモリに読み込んでいます（時間がかかります）...")
# 手術中はメモリを安全に使うため CPU にロードします
model = AutoModelForCausalLM.from_pretrained(
    model_id="/home/tokusagi/tokushusagi/google/gemma-4-E4B-it",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cpu"
)

print("🔪 テンソルの外科手術（スライス）を開始します...")

# 1. Embedding層（入力辞書）のスライス
old_embeddings = model.model.embed_tokens.weight.data
new_embeddings = old_embeddings[keep_ids] # 必要な行だけを物理的に引っこ抜く

model.model.embed_tokens = nn.Embedding(new_vocab_size, model.config.hidden_size, dtype=torch.float16)
model.model.embed_tokens.weight.data = new_embeddings

# 2. LM Head層（出力辞書）のスライス
old_lm_head = model.lm_head.weight.data
new_lm_head = old_lm_head[keep_ids]

model.lm_head = nn.Linear(model.config.hidden_size, new_vocab_size, bias=False, dtype=torch.float16)
model.lm_head.weight.data = new_lm_head

# Configの更新
model.config.vocab_size = new_vocab_size

print("✅ 外科手術完了！モデルをGPUに転送します...")
model = model.to("cuda")

# ==========================================
# フェーズ3: ID変換用マッピングの作成
# ==========================================
# トークナイザーは依然として「古い25万語のID」を出力するため、
# モデルの「新しい圧縮されたID」と翻訳し合う辞書を作ります。
old_to_new_map = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
new_to_old_map = {new_id: old_id for new_id, old_id in enumerate(keep_ids)}

# ==========================================
# フェーズ4: スライス済みモデルでの推論テスト
# ==========================================
print("🚀 推論テストを開始します...")

prompt = "日本の首都はどこですか？"
inputs = tokenizer(prompt, return_tensors="pt")

# 1. 古いIDを「新しいID」に変換してGPUへ送る
# （万が一マッピングにないIDが出た場合は、安全のため未知語トークン[UNK]などにフォールバック）
unk_new_id = old_to_new_map.get(tokenizer.unk_token_id, 0)
mapped_input_ids = [old_to_new_map.get(id.item(), unk_new_id) for id in inputs["input_ids"][0]]
mapped_input_tensor = torch.tensor([mapped_input_ids]).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=mapped_input_tensor,
        max_new_tokens=30,
        pad_token_id=old_to_new_map.get(tokenizer.eos_token_id, 0)
    )

# 2. モデルが出力した「新しいID」を、デコードするために「古いID」に戻す
generated_new_ids = outputs[0].cpu().tolist()
restored_old_ids = [new_to_old_map[nid] for nid in generated_new_ids]

print("\n📝 語彙枝刈りモデルからの返答:")
print(tokenizer.decode(restored_old_ids, skip_special_tokens=True))