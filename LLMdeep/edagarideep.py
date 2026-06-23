import torch
import torch.nn as nn
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 設定エリア
# ==========================================
# 1. 元のモデルのパス（原本の場所）
model_id = "/home/tokusagi/tokushusagi/deepseek"

# 2. 前回のキャリブレーションで保存したスコアファイルのパス
scores_path = "/home/tokusagi/tokushusagi/LLMdeep/expert_scores1.pt" # 環境に合わせて変更してください

# 3. 試したい生存者数（作成したいリストラのターゲット数）
target_sizes = [32, 16, 8]

# ==========================================
# 実行メインブロック
# ==========================================
def main():
    print("📊 スコアデータを読み込んでいます...")
    scores = torch.load(scores_path)

    for target_size in target_sizes:
        print("\n" + "="*50)
        print(f"🔪 枝刈り開始: 【 生存者 {target_size} 人 】の超軽量モデルを作成中...")
        print("="*50)

        # 毎回「元のモデル」を綺麗な状態で読み込む
        # ※枝刈り・保存処理はVRAMを節約するため、device_map="cpu" でメインメモリ上で行うのが安全です
        print("🧠 ベースモデルを読み込んでいます...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            dtype=torch.float16,  # 環境に合わせて変更（torch_dtype から dtype になっています）
            trust_remote_code=True,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)

        # モデルの層リストを取得（環境による変数名の違いを吸収）
        layers = model.model.layers if hasattr(model, "model") else model.layers

        # 各MoE層に対して外科手術を実行
        print("✂️ 各層の専門家とルーターを切除中...")
        for layer_idx, layer_scores in scores.items():
            # Step 2: 生存者リスト（Top-Nの出席番号）を作成
            # 当該層の64人分のスコアから、上位N人のインデックスを取得
            _, top_indices_tensor = torch.topk(layer_scores, target_size)
            top_indices = top_indices_tensor.tolist()
            top_indices.sort()  # 順番を崩さないようにソート（重要）

            # 対象の層を取り出す
            layer = layers[layer_idx]

            # Step 3: 外科手術（物理的な削除）
            
            # ① 専門家（エキスパート）のリストラ
            # 生き残った専門家だけを抽出して、新しいリスト（nn.ModuleList）で上書きする
            old_experts = layer.mlp.experts
            new_experts = nn.ModuleList([old_experts[i] for i in top_indices])
            layer.mlp.experts = new_experts

            # ② ルーター（ゲート）のリストラ
            # ゲートの重み（64行）から、生き残った専門家に対応する行だけを引っこ抜いて上書きする
            old_gate_weight = layer.mlp.gate.weight.data
            layer.mlp.gate.weight = nn.Parameter(old_gate_weight[top_indices, :])
            layer.mlp.gate.out_features = target_size # 設定も書き換え
            
            # ※もしバイアス(偏り)設定があるタイプのルーターなら、それも切り詰める
            if hasattr(layer.mlp.gate, "bias") and layer.mlp.gate.bias is not None:
                layer.mlp.gate.bias = nn.Parameter(layer.mlp.gate.bias.data[top_indices])

        # モデルの設計図（config）の人数も更新しておく
        model.config.n_routed_experts = target_size

        # ★ここに追加！ 最新のTransformersの保存バグを回避するパッチ
        # =======================================================
        for m in model.modules():
            if hasattr(m, "_tied_weights_keys"):
                try:
                    # リストではなく空の辞書で上書きし、エラーをスルーさせる
                    m._tied_weights_keys = {}
                except Exception:
                    pass

        # Step 4: 新しい軽量化モデルとして保存
        output_dir = f"/home/tokusagi/tokushusagi/deepseek_pruned_{target_size}"
        print(f"💾 軽量化モデルを保存中: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # 次のループのためにメモリを徹底的にお掃除（これをしないとメモリ爆発します）
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n🎉🎉 すべてのサイズ (32, 16, 8) の枝刈りモデルが完成しました！！ 🎉🎉")

if __name__ == "__main__":
    main()