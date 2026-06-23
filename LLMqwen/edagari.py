import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------
# 先ほどの枝刈り関数をここに貼り付けます
# ---------------------------------------------------------
import torch

import torch

def prune_qwen_moe(model, expert_scores, num_keep=16):
    """
    算出したスコアに基づき、モデルから不要な専門家を物理的に削除する関数（Qwen3仕様・最終版）
    """
    print(f"各層の専門家を {model.config.num_local_experts}人 から {num_keep}人 に削減します...")
    
    num_layers = model.config.num_hidden_layers
    
    for layer_idx in range(num_layers):
        scores = expert_scores[layer_idx]
        
        # 1. スコアが高い上位 `num_keep` 人のインデックス（背番号）を取得
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)
        _, top_indices = torch.topk(scores.clone().detach(), k=num_keep)
        
        keep_indices = top_indices.tolist()
        keep_indices.sort() # 元の並び順を崩さないように昇順にソート
        
        print(f"第{layer_idx}層 残す専門家: {keep_indices}")
        
        # --- モデルの物理的な切り取り作業 ---
        moe_layer = model.model.layers[layer_idx].mlp
        
        # 2. ルーター（ゲート）の次元を削減
        # ゲートは通常のnn.Linear層なので .weight が必要です
        old_gate = moe_layer.gate
        old_gate.weight.data = old_gate.weight.data[keep_indices, :]
        old_gate.out_features = num_keep
        
        # 3. 専門家（エキスパート）ネットワークの次元を削減
        experts = moe_layer.experts
        
        # ★修正箇所：gate_up_proj と down_proj は直接のパラメータなので .weight を外して直接スライスします
        experts.gate_up_proj.data = experts.gate_up_proj.data[keep_indices, :, :]
        experts.down_proj.data = experts.down_proj.data[keep_indices, :, :]
        
        # バイアスが存在する場合の処理（Qwenには通常ありませんが、安全のための処理も修正）
        if hasattr(experts, "gate_up_proj_bias") and experts.gate_up_proj_bias is not None:
             experts.gate_up_proj_bias.data = experts.gate_up_proj_bias.data[keep_indices, :]
        if hasattr(experts, "down_proj_bias") and experts.down_proj_bias is not None:
             experts.down_proj_bias.data = experts.down_proj_bias.data[keep_indices, :]

    # 4. モデル全体の設定（Config）を更新
    model.config.num_local_experts = num_keep
    
    print("モデルの枝刈り（プルーニング）が完了しました！")
    return model

# ---------------------------------------------------------
# 実行メインブロック
# ---------------------------------------------------------
if __name__ == "__main__":
    model_id = "/home/tokusagi/tokushusagi/Qwen3-30B-A3B" # ※お使いのモデル名
    
    print("モデルをシステムメモリに読み込んでいます...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # ★修正ポイント: device_map="auto" や "cpu" を【完全に削除】します！
    # これによりオフロード機能がオフになり、保存時のエラーが消滅します
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 2. 保存しておいたスコアを読み込む
    print("スコアデータを読み込んでいます...")
    scores = torch.load("expert_scores.pt")
    
    # 3. 枝刈りの実行（128人中、32人を残す）
    pruned_model = prune_qwen_moe(model, scores, num_keep=16)
    
    # 4. 軽量化された新しいモデルを保存
    print("軽量化モデルを保存しています...")
    # 念のための安全策として、現在の状態(state_dict)を明示的に渡して保存します
    pruned_model.save_pretrained("./qwen_pruned_16experts", state_dict=pruned_model.state_dict())
    tokenizer.save_pretrained("./qwen_pruned_16experts")
    
    print("すべて完了しました！お疲れ様でした！")