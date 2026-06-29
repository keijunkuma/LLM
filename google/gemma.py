import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-4-E4B-it"

# ==========================================
# フェーズ1: 必要なIDリストの抽出
# ==========================================
print("🔍 トークナイザーと語彙データを読み込んでいます...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
vocab = tokenizer.get_vocab()

allowed_pattern = re.compile(
    r'^['
    r'\u0000-\u007F'          # 英語 (アルファベット, 数字, 基本記号)
    r'\u3040-\u309F'          # ひらがな
    r'\u30A0-\u30FF'          # カタカナ
    r'\u4E00-\u9FFF'          # 漢字 (CJK統合漢字)
    r'\u3000-\u303F'          # 日本語の句読点（、。など）
    r'\uFF00-\uFFEF'          # 全角英数字・半角カタカナ
    r'\u2581'                 # トークナイザー特有の「空白」を表す記号 ( )
    r']+$'
)
byte_pattern = re.compile(r'^<0x[0-9A-Fa-f]{2}>$')

keep_ids_set = set()

for token_str, token_id in vocab.items():
    if token_id in tokenizer.all_special_ids:
        keep_ids_set.add(token_id)
        continue
    if allowed_pattern.match(token_str):
        keep_ids_set.add(token_id)
        continue
    if byte_pattern.match(token_str):
        keep_ids_set.add(token_id)
        continue

keep_ids = sorted(list(keep_ids_set))
new_vocab_size = len(keep_ids)
print(f"📉 抽出完了！ 語彙サイズ: {tokenizer.vocab_size} ➔ {new_vocab_size}")

# ==========================================
# フェーズ2: モデルのテンソルの物理的スライス
# ==========================================
print("🧠 モデル本体を読み込んでいます...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda"
)

print("🔪 テンソルの外科手術（スライス）を開始します...")
# 1. Embedding層
old_embeddings = model.model.language_model.embed_tokens.weight.data
new_embeddings = old_embeddings[keep_ids]
model.model.language_model.embed_tokens = nn.Embedding(new_vocab_size, model.config.text_config.hidden_size, dtype=torch.float16)
model.model.language_model.embed_tokens.weight.data = new_embeddings

# 2. LM Head層
old_lm_head = model.lm_head.weight.data
new_lm_head = old_lm_head[keep_ids]
model.lm_head = nn.Linear(model.config.text_config.hidden_size, new_vocab_size, bias=False, dtype=torch.float16)
model.lm_head.weight.data = new_lm_head

# Configの更新
model.config.vocab_size = new_vocab_size

print("✅ 外科手術完了！")

# ==========================================
# フェーズ3: ID変換用マッピングの作成
# ==========================================
old_to_new_map = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
new_to_old_map = {new_id: old_id for new_id, old_id in enumerate(keep_ids)}

# ==========================================
# フェーズ4: スライス済みモデルでの推論テスト
# ==========================================
print("🚀 推論テストを開始します...")

prompt = "日本の首都はどこですか？"
inputs = tokenizer(prompt, return_tensors="pt")

unk_new_id = old_to_new_map.get(tokenizer.unk_token_id, 0)
mapped_input_ids = [old_to_new_map.get(id.item(), unk_new_id) for id in inputs["input_ids"][0]]
mapped_input_tensor = torch.tensor([mapped_input_ids]).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=mapped_input_tensor,
        max_new_tokens=30,
        pad_token_id=old_to_new_map.get(tokenizer.eos_token_id, 0)
    )

generated_new_ids = outputs[0].cpu().tolist()
restored_old_ids = [new_to_old_map[nid] for nid in generated_new_ids]

print("\n📝 語彙枝刈りモデルからの返答:")
print(tokenizer.decode(restored_old_ids, skip_special_tokens=True))