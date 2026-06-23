import random

# 元のテンプレートデータ（上記コーパスから抽出）
templates = [
    "こちらは{機関名}でございます。{理由}のため、{行動}をお願いします。",
    "俺なんだけど、{緊急事態}になってさ。すぐ{金額}用意してくれない？",
    "ウイルスに感染しました。{番号}までご連絡ください。"
]

# 変換用の辞書（固有名詞をランダム置換）
entities = {
    "機関名": ["税務署", "警察署", "市役所", "年金事務所", "銀行"],
    "理由": ["還付金手続き", "口座凍結", "料金未納", "保証金必要", "ウイルス感染"],
    "行動": ["ATM へ移動", "電子マネー購入", "振り込み", "画面操作"],
    "緊急事態": ["事故を起こした", "お金を使い込んだ", "薬を買った", "逮捕された"],
    "金額": ["10 万", "50 万", "100 万", "200 万"],
    "番号": ["03-1234-5678", "090-1234-5678", "サポートセンター"]
}

# データ生成ループ
generated_data = []
target_length = 10000  # 目標文字数
current_length = 0

while current_length < target_length:
    template = random.choice(templates)
    text = template
    for key, values in entities.items():
        text = text.replace("{" + key + "}", random.choice(values))
    
    # 学習用タグを付与して悪用防止
    safe_text = f"[学習用データ：詐欺検知] {text}\n"
    generated_data.append(safe_text)
    current_length += len(safe_text)

# 出力
print("".join(generated_data))