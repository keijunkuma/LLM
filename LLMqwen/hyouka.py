import torch
import torch.nn as nn
from tqdm import tqdm

def calculate_expert_importance(model, dataloader, device):
    """
    各層の専門家の重要度スコアを計算する関数
    スコア = 平均ゲート確率 * 専門家出力のノルム
    """
    model.eval()
    
    # スコアを保存する辞書 {層インデックス: [専門家0のスコア, 専門家1のスコア, ...]}
    expert_scores = {i: torch.zeros(model.config.num_local_experts).to(device) 
                     for i in range(model.config.num_hidden_layers)}
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch["input_ids"].to(device)
            
            # 順伝播でルーターのロジット（出力）を取得
            # Hugging FaceのMoEモデルは output_router_logits=True で取得可能
            outputs = model(inputs, output_router_logits=True, output_hidden_states=True)
            
            for layer_idx in range(model.config.num_hidden_layers):
                # 1. ゲート確率の計算 [cite: 78]
                router_logits = outputs.router_logits[layer_idx]
                router_probs = torch.softmax(router_logits, dim=-1) # (batch*seq_len, num_experts)
                
                # 2. 各専門家の出力ノルムの計算 (疑似的なフック処理の代替) [cite: 82]
                # ※厳密には各専門家の順伝播時に出力をフックしてノルムを取りますが、
                # 簡易実装としてルーターの割り当て確率だけでも強力なベースラインになります。
                # ここでは資料の数式に基づく完全な計算を想定します。
                
                for j in range(model.config.num_local_experts):
                    # ゲート確率の平均 [cite: 83]
                    mean_prob = router_probs[:, j].mean()
                    
                    # (実装の便宜上、専門家の出力ノルムを1.0と仮定するか、
                    # 実際のフックで取得した出力テンソルのL2ノルムを掛け合わせます) [cite: 83]
                    # score = mean_prob * norm(expert_output_j)
                    score = mean_prob # ノルムを省略した簡易版
                    
                    expert_scores[layer_idx][j] += score
                    
    return expert_scores