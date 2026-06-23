import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ==========================================
# 1. データセットの準備
# ==========================================
class SagiDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.inputs = []
        print("データセットを読み込んでいます...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # JSONL内の "text" キーの値を使用
                text = data.get("text", "")
                if text:
                    tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
                    self.inputs.append(tokens["input_ids"].squeeze(0))
        print(f"{len(self.inputs)} 件のデータをロードしました。")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx]}

# ==========================================
# 2. フックを用いた完全なスコア計算
# ==========================================
def calculate_expert_importance(model, dataloader, device):
    """
    確率ではなく「実際にTop-Kに選ばれた回数」で
    各専門家の重要度をカウントする関数（最も確実な手法）
    """
    model.eval()
    num_layers = model.config.num_hidden_layers
    num_experts = int(model.config.num_local_experts)
    
    # モデルの設定から、1トークンあたり何人の専門家を選ぶか（Top-K）を取得
    # （設定が見つからない場合は一般的な 8 を使用）
    top_k = getattr(model.config, "num_experts_per_tok", 8)
    
    # スコア（選ばれた回数）を記録する辞書
    expert_scores = {i: torch.zeros(num_experts).to(device) for i in range(num_layers)}
    
    print(f"キャリブレーションを実行中（Top-{top_k} 選択回数ベース）...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch["input_ids"].to(device)
            outputs = model(inputs, output_router_logits=True) 
            
            for layer_idx in range(num_layers):
                # (全トークン数, 専門家数) のロジットを取得
                router_logits = outputs.router_logits[layer_idx]
                
                # ★ここが重要：上位K人の専門家の「インデックス（背番号）」だけを取得
                _, selected_experts = torch.topk(router_logits, k=top_k, dim=-1)
                
                # 今回のバッチの全トークンの中で、各専門家が「何回選ばれたか」を集計
                unique_experts, counts = torch.unique(selected_experts, return_counts=True)
                
                # 回数をスコアとして加算
                for exp_id, count in zip(unique_experts, counts):
                    expert_scores[layer_idx][exp_id.item()] += count.item()

    return expert_scores
# ==========================================
# 3. 実行メインブロック
# ==========================================
if __name__ == "__main__":
    # 環境設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "/home/tokusagi/tokushusagi/tokushusagi" # ※30BクラスのQwen MoEモデルのパスに置き換えてください
    
    # トークナイザーとモデルのロード
    # （実際の30BクラスをロードするにはA100等が複数枚必要です。device_map="auto"を使用）
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda", 
        dtype=torch.float16
    )
    
    # データセットとデータローダーの作成
    dataset = SagiDataset("/home/tokusagi/tokushusagi/sagidataset.jsonl", tokenizer)
    # フック処理の都合上、今回はbatch_size=1で順次処理する構成にしています
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # スコア計算の実行
    scores = calculate_expert_importance(model, dataloader, device)

    # 【追加】計算したスコアをPyTorchのファイル（.pt）として保存する
    torch.save(scores, "expert_scores1.pt")

    # 結果をテキストファイルに書き出して保存する
    with open("mikaesi.txt", "w", encoding="utf-8") as f:
        for i in range(len(scores)):
            # 画面にも表示しつつ...
            print(f"第{i}層の専門家スコア:", scores[i].tolist())
            # ファイルにも同じ内容を書き込む
            f.write(f"第{i}層の専門家スコア: {scores[i].tolist()}\n")