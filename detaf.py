import pandas as pd
import random
import json
import re
from typing import List, Dict

# ================================
# 1. エンティティ辞書（項目をさらに拡充）
# ================================
entities = {
    # 警察官なりすまし用
    "警察署名": ["行徳警察署", "市川警察署", "千葉中央署", "新宿警察署", "サイバー犯罪対策課", "生活安全課", "特殊詐欺対策室"],
    "犯人名": ["ナラオカ", "スズキ", "タナカ", "佐藤", "係長", "タカハシ", "イノウエ"],
    "被害者姓": ["山田", "佐藤", "鈴木", "田中", "高橋", "伊藤", "渡辺", "小林"],
    "職業": ["銀行職員", "郵便局員", "市役所職員", "百貨店の店員", "家電量販店のスタッフ"],
    "人数": ["2", "3", "複数", "4"],
    "犯人役職 1": ["サイトウヨシノリ", "ヤマダナオキ", "スズキケイスケ", "タナカヒロシ"],
    "年齢 1": ["38", "41", "35", "45", "40", "29"],
    "組織 1": ["郵便局", "千葉銀行", "市役所", "信用金庫", "伊勢丹", "ヨドバシカメラ"],
    "犯人役職 2": ["ヤマダナオキ", "タナカヒロシ", "スズキケイスケ", "サイトウヨシノリ"],
    "年齢 2": ["41", "38", "40", "35", "45", "32"],
    "組織 2": ["銀行", "市役所", "業者", "信用金庫", "郵便局", "高島屋"],
    "秘密事項": ["支店名", "詳細", "捜査内容", "相手先の情報", "具体的な場所", "口座の暗証番号"],
    "場所": ["市川", "千葉", "東京", "神奈川", "埼玉", "横浜", "新宿"],
    
    # オレオレ詐欺用
    "息子名前": ["健太", "雄太", "大輔", "拓也", "誠", "翔太", "レン"],
    "孫名前": ["美咲", "優花", "結衣", "愛", "彩"],
    "関係性": ["息子", "孫", "弟", "甥"],
    "緊急理由": ["会社のお金を使い込んでしまった", "事故を起こして相手がヤクザだった", "会社の小切手を電車に忘れた", "痴漢で捕まって示談金が必要", "株で大損して借金取りが来ている"],
    "金額": ["50万円", "100万円", "200万円", "300万円", "500万円", "150万円"],
    "電話番号変更": ["電話番号が変わった", "スマホを落とした", "会社で携帯を変えた", "トイレにスマホを落として水没した"],
    
    # 還付金詐欺用
    "機関名": ["税務署", "市役所", "年金事務所", "保健所", "区役所", "社会保険庁"],
    "理由": ["医療費の過払い", "保険料の払い戻し", "年金の未支給分", "税金の還付", "コロナ補助金の残り", "累積医療費の返還"],
    
    # 架空料金請求用
    "サービス": ["動画サイト", "オンラインゲーム", "SNS", "クラウドサービス", "サブスク", "有料アダルトサイト", "情報通信サイト"],
    
    # 正常会話用
    "銀行名": ["〇〇銀行", "△△銀行", "みずほ銀行", "三菱UFJ銀行", "三井住友銀行", "ゆうちょ銀行"],
    "宅配業者": ["ヤマト運輸", "日本郵便", "佐川急便", "Amazon", "楽天"]
}

# ================================
# 2. 詐欺タイプ別 対話フロー（ご提示いただいた優れたフローをそのまま使用）
# ================================
# ※コードが長くなるため、一部省略して記載していますが、実際にはご提示いただいたフローをそのままここに配置してください。
# (oreore_flow, police_impersonation_flow, refund_flow, fee_request_flow, normal_family_flow, normal_bank_flow)

# 【ここにご提示いただいたフローリストをそのまま挿入します】
# --- 省略部分開始（元のコードのリストを貼り付けてください） ---
oreore_flow = [
    {"speaker": "犯人", "template": "もしもし。"},
    {"speaker": "被害者", "template": "はい。"},
    {"speaker": "犯人", "template": "俺なんだけど。"},
    {"speaker": "被害者", "template": "えっ、誰？"},
    {"speaker": "犯人", "template": "俺、俺。{息子名前}。"},
    {"speaker": "被害者", "template": "あら、{息子名前}？どうしたの？"},
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
    {"speaker": "犯人", "template": "うん、ATMに行って。"},
    {"speaker": "被害者", "template": "わかった、でも電話番号変わったの？"},
    {"speaker": "犯人", "template": "{電話番号変更}んだ。"},
    {"speaker": "被害者", "template": "そう。どこで会えばいい？"},
    {"speaker": "犯人", "template": "口座に振り込んで。"},
    {"speaker": "被害者", "template": "口座番号は？"},
    {"speaker": "犯人", "template": "後でSMSで送る。"},
    {"speaker": "被害者", "template": "わかった、すぐ準備する。"}
]
# （他のフローも同様に定義）
# --- 省略部分終了 ---

# ※テスト実行用ダミーフロー（実際に動かす際は上の省略部分を埋めてください）
police_impersonation_flow = [{"speaker": "犯人", "template": "{警察署名}の{犯人名}です。{被害者姓}さんの口座が狙われています。"}, {"speaker": "被害者", "template": "えっ！"}]
refund_flow = [{"speaker": "犯人", "template": "{機関名}です。{理由}で{金額}の還付があります。ATMへ行ってください。"}, {"speaker": "被害者", "template": "はい。"}]
fee_request_flow = [{"speaker": "犯人", "template": "{サービス}の未納料金{金額}があります。電子マネーで払ってください。"}, {"speaker": "被害者", "template": "ええっ。"}]
normal_family_flow = [{"speaker": "息子", "template": "もしもしお母さん、{息子名前}だけど。日曜帰るよ。"}, {"speaker": "母", "template": "待ってるわ。"}]
normal_bank_flow = [{"speaker": "銀行員", "template": "{銀行名}です。カード更新のお知らせです。パスワードは聞きません。"}, {"speaker": "顧客", "template": "わかりました。"}]


flow_map = {
    "police_impersonation": police_impersonation_flow,
    "oreore": oreore_flow,
    "refund": refund_flow,
    "fee_request": fee_request_flow,
    "normal_family": normal_family_flow,
    "normal_bank": normal_bank_flow
}

trigger_words_map = {
    "police_impersonation": ["警察署", "逮捕", "狙っていた", "口座"],
    "oreore": ["俺なんだけど", "すぐ", "用意して", "誰にも言わないで", "電話番号が変わった", "逮捕されちゃう"],
    "refund": ["還付金", "ATM", "画面の指示", "払い戻し", "至急", "操作をお願いします"],
    "fee_request": ["未納", "法的手続き", "電子マネー", "裁判所", "強制執行"],
    "normal_family": [],
    "normal_bank": ["パスワードをお電話で伺うことはございません"]
}

# ================================
# 3. AI学習用の「理由（CoT）」を動的生成する関数
# ================================
def generate_assistant_reason(fraud_type: str, triggers: List[str], text: str) -> str:
    """抽出されたトリガーワードと詐欺タイプに基づき、AIの模範解答（理由）を生成する"""
    if not triggers and fraud_type.startswith("normal"):
        return "【安全】\n理由: 金銭の要求や不審な誘導、個人情報を聞き出すような発言が含まれておらず、日常的または正規の連絡と判断できます。"

    trigger_str = "」「".join(triggers) if triggers else "不審な言葉"
    
    if fraud_type == "police_impersonation":
        return f"【詐欺】\n理由: 警察官を名乗り、「{trigger_str}」などの言葉で被害者の不安を煽っています。警察が電話で口座情報や暗証番号を聞き出したり、キャッシュカードを預かることは絶対にありません。典型的な警察官なりすまし詐欺です。"
    elif fraud_type == "oreore":
        return f"【詐欺】\n理由: 息子などの親族を装い、「{trigger_str}」などと言って電話番号が変わったと思い込ませ、緊急で金銭（ATM振込など）を要求しています。典型的なオレオレ詐欺の兆候が完全に一致しています。"
    elif fraud_type == "refund":
        return f"【詐欺】\n理由: 公的機関を名乗り還付金の手続きと称して、「{trigger_str}」へ誘導しています。役所がATMの操作で還付金を返還することはシステム上あり得ません。還付金詐欺です。"
    elif fraud_type == "fee_request":
        return f"【詐欺】\n理由: 身に覚えのない未納料金を口実に「{trigger_str}」という言葉で脅迫し、電子マネーでの支払いを要求しています。典型的な架空料金請求詐欺の手口です。"
    else:
        return "【詐欺】\n理由: 会話内に特殊詐欺特有の誘導が含まれています。"

# ================================
# 4. テンプレート処理・データ生成の中核
# ================================
def fill_template(template: str, entities: Dict, context: Dict = None) -> str:
    text = template
    if context is None:
        context = {}
        
    for key, values in entities.items():
        placeholder = "{" + key + "}"
        if placeholder in text:
            # 一度選んだ値はコンテキストに保存し、同一会話内では同じ値を使い回す（一貫性）
            if key not in context:
                context[key] = random.choice(values)
            text = text.replace(placeholder, context[key])
    return text

def generate_training_record(fraud_type: str, entities: Dict) -> Dict:
    flow = flow_map.get(fraud_type, police_impersonation_flow)
    dialogue_lines = []
    context = {} # この会話内でのみ共有するエンティティの記憶
    
    # オレオレ詐欺と家族会話の名前の一貫性を担保
    if fraud_type in ["oreore", "normal_family"]:
        context["息子名前"] = random.choice(entities["息子名前"])

    for turn in flow:
        speaker = turn["speaker"]
        text = fill_template(turn["template"], entities, context)
        dialogue_lines.append(f"[{speaker}]: {text}")
        
    full_text = "\n".join(dialogue_lines)
    
    # トリガーワードの抽出と理由の生成
    triggers = trigger_words_map.get(fraud_type, [])
    found_triggers = [word for word in triggers if word in full_text][:3] # 最大3つまで抽出
    
    assistant_reason = generate_assistant_reason(fraud_type, found_triggers, full_text)
    
    return {
        "system": "あなたは特殊詐欺の専門判定AIです。会話内容を分析し、特殊詐欺の可能性を判定してください。必ず【詐欺】か【安全】を出力し、続けてその理由を説明してください。",
        "user": full_text,
        "assistant": assistant_reason
    }

# ================================
# 5. データセット大量生成とCSV保存
# ================================
def generate_large_dataset(target_size=1000, fraud_ratio=0.7):
    print(f"高品質な学習データを {target_size} 件生成しています...")
    dataset = []
    fraud_types = ["police_impersonation", "oreore", "refund", "fee_request"]
    normal_types = ["normal_family", "normal_bank"]
    
    fraud_count = int(target_size * fraud_ratio)
    normal_count = target_size - fraud_count
    
    for _ in range(fraud_count):
        fraud_type = random.choice(fraud_types)
        dataset.append(generate_training_record(fraud_type, entities))
        
    for _ in range(normal_count):
        normal_type = random.choice(normal_types)
        dataset.append(generate_training_record(normal_type, entities))
        
    random.shuffle(dataset) # 詐欺と安全をランダムに混ぜる
    
    # CSV形式に変換して保存
    df = pd.DataFrame(dataset)
    filename = "training_data_advanced.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    
    print(f"\n生成完了! '{filename}' に保存しました。")
    print(f"詐欺データ: {fraud_count} 件 / 安全データ: {normal_count} 件")
    
    # サンプルを1つ表示
    print("\n--- 【生成サンプル】 ---")
    print(f"[User Input]\n{df.iloc[0]['user']}\n")
    print(f"[AI Output]\n{df.iloc[0]['assistant']}")
    print("------------------------")

if __name__ == "__main__":
    # ここで欲しいデータ件数を指定します
    generate_large_dataset(target_size=1000, fraud_ratio=0.75)