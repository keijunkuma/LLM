import random
import json
from typing import List, Dict

# ================================
# 1. エンティティ辞書（名前を一貫性を持たせる）
# ================================

entities = {
    # 警察官なりすまし用
    "警察署名": ["行徳警察署", "市川警察署", "千葉中央署", "〇〇警察署", "生活安全課"],
    "犯人名": ["ナラオカ", "スズキ", "タナカ", "佐藤", "係長", "担当"],
    "被害者姓": ["山田", "佐藤", "鈴木", "田中", "高橋", "伊藤"],
    "職業": ["銀行職員", "郵便局員", "市役所職員", "業者", "金融機関員"],
    "人数": ["2", "3", "複数"],
    "犯人役職 1": ["サイトウヨシノリ", "ヤマダナオキ", "スズキケイスケ", "タナカヒロシ"],
    "年齢 1": ["38", "41", "35", "45", "40"],
    "組織 1": ["郵便局", "千葉銀行", "市役所", "信用金庫", "銀行"],
    "犯人役職 2": ["ヤマダナオキ", "タナカヒロシ", "スズキケイスケ", "サイトウヨシノリ"],
    "年齢 2": ["41", "38", "40", "35", "45"],
    "組織 2": ["銀行", "市役所", "業者", "信用金庫", "郵便局"],
    "秘密事項": ["支店名", "詳細", "捜査内容", "相手先の情報", "具体的な場所"],
    "場所": ["市川", "千葉", "東京", "神奈川", "埼玉"],
    
    # オレオレ詐欺用（名前を一貫）
    "息子名前": ["健太", "雄太", "大輔", "拓也", "誠"],
    "孫名前": ["美咲", "優花", "結衣", "愛", "彩"],
    "関係性": ["息子", "孫", "弟", "甥"],
    "緊急理由": ["会社のお金を使い込んでしまった", "事故を起こしてしまった", "薬を買ってしまった", "逮捕されてしまった", "借金をしてしまった"],
    "金額": ["50 万円", "100 万円", "200 万円", "300 万円", "500 万円"],
    "電話番号変更": ["電話番号が変わった", "スマホを落とした", "会社で番号を変えた", "新しい番号になった"],
    
    # 還付金詐欺用
    "機関名": ["税務署", "市役所", "年金事務所", "保健所", "区役所"],
    "理由": ["医療費の過払い", "保険料の払い戻し", "年金の未支給分", "税金の還付", "補助金"],
    
    # 架空料金請求用
    "サービス": ["動画サイト", "オンラインゲーム", "SNS", "クラウドサービス", "サブスク"],
    
    # 正常会話用
    "銀行名": ["〇〇銀行", "△△銀行", "××銀行", "みずほ銀行", "三菱 UFJ 銀行"],
    "宅配業者": ["ヤマト運輸", "日本郵便", "佐川急便", "Amazon", "楽天"]
}

# ================================
# 2. 詐欺タイプ別 対話フロー（名前の一貫性を確保）
# ================================

# オレオレ詐欺フロー（修正版）
oreore_flow = [
    {"speaker": "犯人", "template": "もしもし。"},
    {"speaker": "被害者", "template": "はい。"},
    {"speaker": "犯人", "template": "俺なんだけど。"},
    {"speaker": "被害者", "template": "えっ、誰？"},
    {"speaker": "犯人", "template": "俺、俺。{息子名前}。"},
    {"speaker": "被害者", "template": "あら、{息子名前}？どうしたの？"},  # 同じ変数を使用
    {"speaker": "犯人", "template": "ちょっと大事なことになってさ。"},
    {"speaker": "被害者", "template": "えっ、どうしたの？"},
    {"speaker": "犯人", "template": "{緊急理由}。"},
    {"speaker": "被害者", "template": "えっ、警察？どうして？"},
    {"speaker": "犯人", "template": "だから今話してる時間がないの。"},
    {"speaker": "被害者", "template": "ちょっと待って、お父さんに聞くね。"},
    {"speaker": "犯人", "template": "だめ！絶対言わないで！"},
    {"speaker": "被害者", "template": "どうして？"},
    {"speaker": "犯人", "template": "バレたら逮捕されちゃうから！"},
    {"speaker": "被害者", "template": "でも、いくら必要なの？"},
    {"speaker": "犯人", "template": "すぐ{金額}用意してくれない？"},
    {"speaker": "被害者", "template": "キャッシュカードで？"},
    {"speaker": "犯人", "template": "うん、ATM に行って。"},
    {"speaker": "被害者", "template": "わかった、でも電話番号変わったの？"},
    {"speaker": "犯人", "template": "{電話番号変更}んだ。"},
    {"speaker": "被害者", "template": "そう。どこで会えばいい？"},
    {"speaker": "犯人", "template": "口座に振り込んで。"},
    {"speaker": "被害者", "template": "口座番号は？"},
    {"speaker": "犯人", "template": "後で SMS で送る。"},
    {"speaker": "被害者", "template": "わかった、すぐ準備する。"}
]

# 警察官なりすましフロー（PDF に基づく）
police_impersonation_flow = [
    {"speaker": "犯人", "template": "恐れ入ります。{警察署名} 生活安全課の{犯人名}と申します。"},
    {"speaker": "被害者", "template": "はい。"},
    {"speaker": "犯人", "template": "{被害者姓}さんのご自宅でお間違いないでしょうか？"},
    {"speaker": "被害者", "template": "そうです。"},
    {"speaker": "犯人", "template": "あっ、いきなりのご連絡で申し訳ないんですが、{被害者姓}さんにお伺いしたいことがあって。"},
    {"speaker": "被害者", "template": "はい。"},
    {"speaker": "犯人", "template": "あのー、今ですね、うちの署の方で{職業}を{人数}名逮捕してまして。"},
    {"speaker": "被害者", "template": "はい。"},
    {"speaker": "犯人", "template": "この二人がですね、{被害者姓}さんのご自宅を狙っていたと言っているものですからね。"},
    {"speaker": "被害者", "template": "えっ、そうなんですか。"},
    {"speaker": "犯人", "template": "ちょっと名前の確認を取っていただきたいんですけど。"},
    {"speaker": "被害者", "template": "...はい。"},
    {"speaker": "犯人", "template": "あのー、{犯人役職 1}という{年齢 1}歳の{組織 1}の職員と。"},
    {"speaker": "被害者", "template": "はい。"},
    {"speaker": "犯人", "template": "{犯人役職 2}という{年齢 2}歳の{組織 2}の職員ってご存じないですかね？"},
    {"speaker": "被害者", "template": "えー、わかんないー、どこの支店ですか？"},
    {"speaker": "犯人", "template": "えーと、これ{場所}市内なんですけれども、ちょっと{秘密事項}とかはね、お伝えすることができないんですけど。"},
    {"speaker": "被害者", "template": "特にはないー、あのー名前ってどこにでもありそうな名前なんで。"},
    {"speaker": "犯人", "template": "あー、そうですか。じゃあ分からないですかね。"},
    {"speaker": "被害者", "template": "わからないですねー。"},
    {"speaker": "犯人", "template": "あー、そうですか。ちょっとご家族の方いらっしゃったらですね、この名前知ってるかどうかちょっと聞いてもらえます？"},
    {"speaker": "被害者", "template": "えっと、すいません、えーっと、今お電話ってことなんですけども。"},
    {"speaker": "犯人", "template": "はいはい。"},
    {"speaker": "被害者", "template": "すいません、かけ直していいですか。どこにお電話、{警察署名}ですよね？"},
    {"speaker": "犯人", "template": "はい、はい。"},
    {"speaker": "被害者", "template": "あっ、教えてもらえる？"},
    {"speaker": "被害者", "template": "（電話断）"}
]

# 還付金詐欺フロー
refund_flow = [
    {"speaker": "犯人", "template": "こちら{機関名}でございます。"},
    {"speaker": "被害者", "template": "はい。"},
    {"speaker": "犯人", "template": "{被害者姓}様のお宅でお間違いないでしょうか。"},
    {"speaker": "被害者", "template": "そうです。"},
    {"speaker": "犯人", "template": "実はですね、{理由}の還付金がございます。"},
    {"speaker": "被害者", "template": "還付金ですか？"},
    {"speaker": "犯人", "template": "はい、{金額}の払い戻し手続きが必要です。"},
    {"speaker": "被害者", "template": "どうすればいいですか？"},
    {"speaker": "犯人", "template": "お手数ですが、近くの ATM へ移動していただきます。"},
    {"speaker": "被害者", "template": "ATM ですか？"},
    {"speaker": "犯人", "template": "はい、画面の指示に従って操作をお願いします。"},
    {"speaker": "被害者", "template": "どのボタンを押せば？"},
    {"speaker": "犯人", "template": "まず、振込メニューを選択してください。"},
    {"speaker": "被害者", "template": "振込？"},
    {"speaker": "犯人", "template": "はい、還付金受取コースとなります。"},
    {"speaker": "被害者", "template": "そんなコースありましたっけ。"},
    {"speaker": "犯人", "template": "今回の特別措置でございます。"},
    {"speaker": "被害者", "template": "で、次は？"},
    {"speaker": "犯人", "template": "暗証番号を入力してください。"},
    {"speaker": "被害者", "template": "暗証番号も？"},
    {"speaker": "犯人", "template": "はい、本人確認のためです。"},
    {"speaker": "被害者", "template": "わかりました。"},
    {"speaker": "犯人", "template": "では、振込先を入力します。"},
    {"speaker": "被害者", "template": "はい。"},
    {"speaker": "犯人", "template": "指示通りに入力してください。"},
    {"speaker": "被害者", "template": "（電話断）"}
]

# 架空料金請求フロー
fee_request_flow = [
    {"speaker": "犯人", "template": "【重要】{サービス}からのお知らせです。"},
    {"speaker": "被害者", "template": "はい。"},
    {"speaker": "犯人", "template": "{被害者姓}様、利用料金が未納となっております。"},
    {"speaker": "被害者", "template": "未納？"},
    {"speaker": "犯人", "template": "はい、{金額}のお支払いが必要です。"},
    {"speaker": "被害者", "template": "心当たりないんですが。"},
    {"speaker": "犯人", "template": "システム上の記録がございます。"},
    {"speaker": "被害者", "template": "いつの利用ですか？"},
    {"speaker": "犯人", "template": "先月分の利用料金です。"},
    {"speaker": "被害者", "template": "でも使ってないですよ。"},
    {"speaker": "犯人", "template": "本日中に決済がない場合、法的手続きに移行します。"},
    {"speaker": "被害者", "template": "法的手続き？"},
    {"speaker": "犯人", "template": "はい、裁判所へ通知いたします。"},
    {"speaker": "被害者", "template": "どうすればいいですか？"},
    {"speaker": "犯人", "template": "コンビニで電子マネーを購入し、番号をお送りください。"},
    {"speaker": "被害者", "template": "電子マネー？"},
    {"speaker": "犯人", "template": "はい、iTunes カードまたは Google Play カードです。"},
    {"speaker": "被害者", "template": "いくら分？"},
    {"speaker": "犯人", "template": "{金額}分です。"},
    {"speaker": "被害者", "template": "番号はどうするの？"},
    {"speaker": "犯人", "template": "SMS でお送りください。"},
    {"speaker": "被害者", "template": "わかりました。"},
    {"speaker": "犯人", "template": "至急お願いします。"},
    {"speaker": "被害者", "template": "（電話断）"}
]

# 正常な家族会話フロー
normal_family_flow = [
    {"speaker": "息子", "template": "もしもし、お母さん？"},
    {"speaker": "母", "template": "あら、{息子名前}？どうしたの？"},
    {"speaker": "息子", "template": "今週の日曜日、帰ろうと思うんだけど。"},
    {"speaker": "母", "template": "あら嬉しい。何時頃来るの？"},
    {"speaker": "息子", "template": "お昼過ぎくらいに。"},
    {"speaker": "母", "template": "わかった。晩飯何がいい？"},
    {"speaker": "息子", "template": "ハンバーグがいいな。"},
    {"speaker": "母", "template": "わかった。買い出しに行ってくるね。"},
    {"speaker": "息子", "template": "ありがとう。"},
    {"speaker": "母", "template": "天気予報だと雨かもよ。"},
    {"speaker": "息子", "template": "傘持ってくるね。"},
    {"speaker": "母", "template": "気をつけて来てね。"},
    {"speaker": "息子", "template": "わかった。"},
    {"speaker": "母", "template": "お父さんも喜ぶわよ。"},
    {"speaker": "息子", "template": "お父さん元気？"},
    {"speaker": "母", "template": "元気よ、ゴルフ行ってる。"},
    {"speaker": "息子", "template": "そう、じゃあ日曜日に。"},
    {"speaker": "母", "template": "はい、待ってるね。"},
    {"speaker": "息子", "template": "じゃあね。"},
    {"speaker": "母", "template": "さようなら。"}
]

# 正常な銀行会話フロー
normal_bank_flow = [
    {"speaker": "銀行員", "template": "いつもご利用いただきありがとうございます。{銀行名}です。"},
    {"speaker": "顧客", "template": "はい。"},
    {"speaker": "銀行員", "template": "{被害者姓}様のお宅でお間違いないでしょうか。"},
    {"speaker": "顧客", "template": "そうです。"},
    {"speaker": "銀行員", "template": "カードの更新時期となりましたので。"},
    {"speaker": "顧客", "template": "あ、そうですか。"},
    {"speaker": "銀行員", "template": "新しいカードを郵送予定です。"},
    {"speaker": "顧客", "template": "いつ頃届きますか？"},
    {"speaker": "銀行員", "template": "来週中には届く予定です。"},
    {"speaker": "顧客", "template": "わかりました。"},
    {"speaker": "銀行員", "template": "届きましたら、裏面の署名をお願いいたします。"},
    {"speaker": "顧客", "template": "はい。"},
    {"speaker": "銀行員", "template": "なお、パスワードをお電話で伺うことはございません。"},
    {"speaker": "顧客", "template": "そうですか。"},
    {"speaker": "銀行員", "template": "ご注意ください。"},
    {"speaker": "顧客", "template": "はい。"},
    {"speaker": "銀行員", "template": "他にご用件はございますか？"},
    {"speaker": "顧客", "template": "ありません。"},
    {"speaker": "銀行員", "template": "ありがとうございます。"},
    {"speaker": "顧客", "template": "こちらこそ。"}
]

# ================================
# 3. フロー定義マッピング
# ================================

flow_map = {
    "police_impersonation": police_impersonation_flow,
    "oreore": oreore_flow,
    "refund": refund_flow,
    "fee_request": fee_request_flow,
    "normal_family": normal_family_flow,
    "normal_bank": normal_bank_flow
}

# ================================
# 4. トリガーワード定義
# ================================

trigger_words_map = {
    "police_impersonation": ["警察署", "逮捕", "狙っていた", "お伝えできません", "捜査中", "ご家族の方"],
    "oreore": ["俺なんだけど", "すぐ", "用意して", "誰にも言わないで", "電話番号が変わった", "警察が来てる"],
    "refund": ["還付金", "ATM", "画面の指示", "払い戻し", "至急", "操作をお願いします"],
    "fee_request": ["未納", "法的手続き", "電子マネー", "裁判所", "強制執行", "最終通知"],
    "normal_family": [],
    "normal_bank": ["パスワードをお電話で伺うことはございません"]
}

# ================================
# 5. テンプレート充填関数（1 回の生成で全置換）
# ================================

def fill_template(template: str, entities: Dict) -> str:
    """テンプレートのプレースホルダーをランダムな値で置換"""
    text = template
    for key, values in entities.items():
        placeholder = "{" + key + "}"
        if placeholder in text:
            # 同じテンプレート内では同じ値を使用
            text = text.replace(placeholder, random.choice(values))
    return text

# ================================
# 6. 一貫性チェック関数
# ================================

def check_name_consistency(dialogue: List[Dict]) -> bool:
    """名前の一貫性をチェック"""
    # オレオレ詐欺の場合、犯人が名乗った名前と被害者が呼ぶ名前が一致するか確認
    fraud_name = None
    victim_name = None
    
    for turn in dialogue:
        text = turn["text"]
        speaker = turn["speaker"]
        
        # 犯人が名乗る名前を抽出
        if speaker == "犯人" and "俺、俺。" in text:
            import re
            match = re.search(r"俺、俺。\s*(\w+)", text)
            if match:
                fraud_name = match.group(1)
        
        # 被害者が呼ぶ名前を抽出
        if speaker == "被害者" and "あら、" in text and "？" in text:
            import re
            match = re.search(r"あら、(\w+)？", text)
            if match:
                victim_name = match.group(1)
    
    # 両方抽出できて不一致なら False
    if fraud_name and victim_name and fraud_name != victim_name:
        return False
    
    return True

# ================================
# 7. 会話データ生成関数（修正版）
# ================================

def generate_conversation(fraud_type: str, entities: Dict) -> Dict:
    """1 つの会話データを生成（20 往復）"""
    flow = flow_map.get(fraud_type, police_impersonation_flow)
    
    # 名前一貫性の確保（オレオレ詐欺の場合）
    if fraud_type == "oreore":
        # 事前に名前を 1 つ選択し、エンティティに設定
        selected_name = random.choice(entities["息子名前"])
        # 会話中ずっと同じ名前を使用するために、テンプレート処理を個別に行う
        pass
    
    dialogue_lines = []
    full_text = ""
    
    # 事前に名前を決定（オレオレ詐欺用）
    pre_selected_name = None
    if fraud_type == "oreore":
        pre_selected_name = random.choice(entities["息子名前"])
    elif fraud_type == "normal_family":
        pre_selected_name = random.choice(entities["息子名前"])
    
    for turn in flow:
        speaker = turn["speaker"]
        template = turn["template"]
        
        # オレオレ詐欺と正常家族会話の名前を一貫させる
        if fraud_type in ["oreore", "normal_family"] and pre_selected_name:
            # テンプレート内の {息子名前} を事前に選択した名前で置換
            text = template.replace("{息子名前}", pre_selected_name)
            # その他のプレースホルダーも置換
            text = fill_template(text, entities)
        else:
            text = fill_template(template, entities)
        
        dialogue_lines.append({
            "speaker": speaker,
            "text": text
        })
        
        full_text += f"{speaker}: {text}\n"
    
    # 一貫性チェック
    is_consistent = check_name_consistency(dialogue_lines)
    
    # トリガーワード抽出
    triggers = extract_trigger_words(full_text, fraud_type)
    
    # ラベル設定
    is_fraud = not fraud_type.startswith("normal")
    label = {
        "is_fraud": is_fraud,
        "fraud_type": fraud_type,
        "risk_level": "high" if is_fraud else "low",
        "trigger_words": triggers,
        "turn_count": len(dialogue_lines),
        "source_domain": "phone_call",
        "name_consistent": is_consistent
    }
    
    return {
        "text": full_text,
        "dialogue": dialogue_lines,
        "label": label
    }

# ================================
# 8. トリガーワード抽出関数
# ================================

def extract_trigger_words(text: str, fraud_type: str) -> List[str]:
    """テキスト内容から実際に含まれるトリガーワードを抽出"""
    if fraud_type.startswith("normal"):
        return []
    
    triggers = trigger_words_map.get(fraud_type, [])
    found_triggers = [word for word in triggers if word in text]
    return found_triggers[:5]

# ================================
# 9. データセット生成メイン関数
# ================================

def generate_dataset(target_conversations: int = 50, fraud_ratio: float = 0.7) -> List[Dict]:
    """データセットを生成"""
    dataset = []
    
    fraud_types = ["police_impersonation", "oreore", "refund", "fee_request"]
    normal_types = ["normal_family", "normal_bank"]
    
    fraud_count = int(target_conversations * fraud_ratio)
    normal_count = target_conversations - fraud_count
    
    # 詐欺データ生成
    for i in range(fraud_count):
        fraud_type = random.choice(fraud_types)
        record = generate_conversation(fraud_type, entities)
        
        # 名前の一貫性がなければ再生成
        attempts = 0
        while not record["label"]["name_consistent"] and attempts < 3:
            record = generate_conversation(fraud_type, entities)
            attempts += 1
        
        dataset.append(record)
    
    # 正常データ生成
    for i in range(normal_count):
        normal_type = random.choice(normal_types)
        record = generate_conversation(normal_type, entities)
        
        # 名前の一貫性がなければ再生成
        attempts = 0
        while not record["label"]["name_consistent"] and attempts < 3:
            record = generate_conversation(normal_type, entities)
            attempts += 1
        
        dataset.append(record)
    
    # シャッフル
    random.shuffle(dataset)
    
    return dataset

# ================================
# 10. 実行と出力
# ================================

if __name__ == "__main__":
    # データセット生成
    dataset = generate_dataset(target_conversations=50, fraud_ratio=0.7)
    
    # 最初の 2 件を詳細表示
    print("=" * 80)
    print("生成された会話データセット（サンプル 2 件）")
    print("=" * 80)
    
    for i, record in enumerate(dataset[:2], 1):
        print(f"\n【サンプル {i}】")
        print(f"タイプ：{record['label']['fraud_type']}")
        print(f"詐欺フラグ：{record['label']['is_fraud']}")
        print(f"ターン数：{record['label']['turn_count']}")
        print(f"名前一貫性：{record['label']['name_consistent']}")
        print(f"トリガーワード：{record['label']['trigger_words']}")
        print("-" * 80)
        print(record['text'])
        print("-" * 80)
    
    # 統計情報
    print("\n" + "=" * 80)
    print("統計情報")
    print("=" * 80)
    
    fraud_count = sum(1 for r in dataset if r['label']['is_fraud'])
    normal_count = len(dataset) - fraud_count
    consistent_count = sum(1 for r in dataset if r['label']['name_consistent'])
    
    print(f"総会話数：{len(dataset)}")
    print(f"詐欺データ：{fraud_count}件 ({fraud_count/len(dataset)*100:.1f}%)")
    print(f"正常データ：{normal_count}件 ({normal_count/len(dataset)*100:.1f}%)")
    print(f"名前一貫性 OK: {consistent_count}件 ({consistent_count/len(dataset)*100:.1f}%)")
    print(f"総ターン数：{sum(r['label']['turn_count'] for r in dataset)}")
    print(f"総単語数：{sum(len(r['text'].split()) for r in dataset)}")
    
    # タイプ別内訳
    type_counts = {}
    for r in dataset:
        t = r['label']['fraud_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\nタイプ別内訳：")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}件")
    
    # JSONL 形式で保存
    with open("fraud_conversation_data_fixed.jsonl", "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\nデータを fraud_conversation_data_fixed.jsonl として保存しました。")