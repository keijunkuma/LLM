import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ==========================================
# 1. データセットの準備 (変更なし)
# ==========================================
class SagiDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.inputs = []
        print("データセットを読み込んでいます...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
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
# 2. フックを用いた完全なスコア計算（横取り特化版）
# ==========================================
def calculate_expert_importance(model, dataloader, device):
    """
    確率ではなく「実際にTop-Kに選ばれた回数」で
    各専門家の重要度をカウントする関数（DeepSeekMoE対応版）
    """
    model.eval()
    
    # ★変更点1: DeepSeekのConfig変数名に対応
    # 共有専門家(Shared)は常に選ばれるので除外し、選択対象となる「ルーティング専門家(Routed)」の数を取得
    num_experts = getattr(model.config, "n_routed_experts", 64) 
    top_k = getattr(model.config, "num_experts_per_tok", 6)
    
    # スコアを記録する辞書（ここではMoE層のインデックスをキーにするため、後で動的に作成します）
    expert_scores = {}
    
    # ルーターの計算結果を横取りして保存する辞書
    router_logits_dict = {}

    # ----------------------------------------------------
    # ★追加：マイク（フック）の定義と設置
    # ----------------------------------------------------
    def get_hook(layer_idx):
        def hook(module, input, output):
            # ★変更：DeepSeekのルーターは (topk_idx, topk_weight, aux_loss) の3つを返す！
            # output[0] が「既に選ばれたTop-Kの専門家インデックス(topk_idx)」なので、それだけを横取りする
            router_logits_dict[layer_idx] = output[0].detach()
        return hook

    handles = []
    # モデル内のすべての層をチェックし、MoE層の「ルーター（gate）」にマイクを仕掛ける
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for i, layer in enumerate(layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            handle = layer.mlp.gate.register_forward_hook(get_hook(i))
            handles.append(handle)
    # ----------------------------------------------------

    # 実行する前に、モデル全体の設定として「ルーターのログを出す」をオンにしておく
    model.config.output_router_logits = True
    model.config.use_cache = False

    print(f"キャリブレーションを実行中（Top-{top_k} 選択回数ベース）...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch["input_ids"].to(device)
            
            # バッチごとに横取り用の辞書をリセット
            router_logits_dict.clear()

            # 引数から output_router_logits=True を削除（エラー回避）
            outputs = model(inputs)
            
            # ★変更点2: outputs.router_logitsの仕様バグを回避するため、
            # フックで横取りしたルーターのログ（router_logits_dict）を使ってループする。
            # DeepSeekは最初の数層がDense層（専門家なし）だが、マイクはMoE層にしか
            # 仕掛けていないため、ルーターが存在する層のみ安全に処理される。
            if not router_logits_dict:
                continue
                
            for moe_layer_idx, selected_experts in router_logits_dict.items():
                # 辞書の初期化（最初のバッチ時のみ）
                if moe_layer_idx not in expert_scores:
                    expert_scores[moe_layer_idx] = torch.zeros(num_experts).to(device)

                # ルーター自身が既にTop-Kの専門家を選んでくれているため、自前での計算は不要に！
                #_, selected_experts = torch.topk(router_logits, k=top_k, dim=-1)
                
                # 今回のバッチで各専門家が「何回選ばれたか」を集計
                unique_experts, counts = torch.unique(selected_experts, return_counts=True)
                
                # 回数をスコアとして加算
                for exp_id, count in zip(unique_experts, counts):
                    expert_scores[moe_layer_idx][exp_id.item()] += count.item()

    # 終わったらマイク（フック）を取り外す（メモリリーク防止）
    for handle in handles:
        handle.remove()

    return expert_scores

# ==========================================
# 3. 実行メインブロック (★読み込み部分を変更)
# ==========================================
if __name__ == "__main__":
    # 環境設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "/home/tokusagi/tokushusagi/deepseek" 
    
    # ★変更点3: trust_remote_code=True と torch_dtype に修正
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda", 
        dtype=torch.float16,  # dtype -> torch_dtype に修正
        trust_remote_code=True      # DeepSeekのカスタムコード実行を許可
    )
    
    dataset = SagiDataset("/home/tokusagi/tokushusagi/sagidataset.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # スコア計算の実行
    scores = calculate_expert_importance(model, dataloader, device)

    # 計算したスコアをPyTorchのファイル（.pt）として保存する
    torch.save(scores, "expert_scores1.pt")

    # 結果をテキストファイルに書き出して保存する
    with open("mikaesi.txt", "w", encoding="utf-8") as f:
        # 辞書のキー(MoE層のインデックス)順にソートして出力
        for layer_idx in sorted(scores.keys()):
            score_list = scores[layer_idx].tolist()
            print(f"MoE層 {layer_idx} の専門家スコア:", score_list)
            f.write(f"MoE層 {layer_idx} の専門家スコア: {score_list}\n")