# ==========================================
# 準備：必要な道具（ライブラリ）を読み込む
# ==========================================
import torch                                 # AIの根幹である行列計算やテンソル（多次元配列）を扱うための基本ライブラリ「PyTorch」
import pandas as pd                          # CSVファイルなどの表計算データを読み込んだり操作したりするためのライブラリ
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments # Hugging Faceの基本ツール。辞書(Tokenizer)、AI本体(Model)、学習ルール(Arguments)
from peft import LoraConfig, get_peft_model  # 省メモリ学習の要「LoRA（バイパス回路）」の設定と、AIにそれを取り付けるためのツール
from datasets import Dataset                 # データをAIが効率よく読み込める専用の形式に変換するためのツール
from trl import SFTTrainer,SFTConfig                   # 教師あり学習（SFT: Supervised Fine-Tuning）を簡単に実行するための専用トレーナー

# ==========================================
# 1. 圧縮なし(16bit)で高品質に読み込む
# ==========================================
# ベースとなる枝刈り済みモデル（16 expertsに減らして混乱している脳）の保存場所を指定
model_path = "./qwen_pruned_16experts"

print("モデルを16bit精度で読み込んでいます...")

# AIが人間の言葉を理解できる数字（トークン）に変換するための辞書（Tokenizer）を読み込む
tokenizer = AutoTokenizer.from_pretrained(model_path)

# AIの脳みそ（モデル）本体を読み込む
model = AutoModelForCausalLM.from_pretrained(
    model_path,                 # 読み込むモデルのパス
    device_map="auto",          # DGXの複数GPU（A100など）の空いている場所に自動で上手く振り分けて配置する設定
    torch_dtype=torch.bfloat16  # 【重要】4bit圧縮をせず、最新GPU向けの超高精度な16bit（bfloat16）で生のまま読み込む
)

# ==========================================
# 2. フルパワーLoRAの設定（表現力大幅アップ）
# ==========================================
# 枝刈りで壊れた脳を直接書き換えるのではなく、「横に賢いバイパス回路（LoRA）」を取り付けて再教育する設定
peft_config = LoraConfig(
    r=64,               # バイパス回路の「太さ（Rank）」。16から64へ4倍に拡大し、複雑な詐欺の言い回しも記憶できるようにする
    lora_alpha=128,     # 学習した内容をベースの脳に「どれくらいの強さで反映させるか（倍率）」。基本はrの2倍にするのが鉄則
    # 【重要】AIの思考回路(q,k,v,o)と、枝刈りした専門家(gate,up,down)の【すべて】にバイパスを接続し、脳全体を治療する
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,  # 学習中にランダムに5%の神経を休ませることで、丸暗記（過学習）を防ぎ、応用力をつけさせる
    bias="none",        # バイアス（偏りパラメータ）は学習させない（一般的な言語モデル学習の推奨設定）
    task_type="CAUSAL_LM" # このAIが「前の言葉から次の言葉を予測するAI（因果的言語モデル）」であることを指定
)

# 上記の設定（バイパス回路）を、実際にAIの脳に取り付ける（合体させる）
model = get_peft_model(model, peft_config)

# 脳全体のうち、「どのくらいの割合（パラメータ数）を学習させるか」を画面に表示して確認する
model.print_trainable_parameters()

# ==========================================
# 3. データセットの準備（CSV読み込み）
# ==========================================
print("CSVから学習データを読み込みます...")

# 先ほど自動生成した何百件もの高品質な詐欺＆安全会話データ（CSV）を読み込む
df = pd.read_csv("/home/tokusagi/tokushusagi/training_data_advanced.csv")

# CSVの「system(役割)」「user(入力)」「assistant(正解の出力)」を、Qwenが理解できるチャットのフォーマットに結合する関数
def format_chat(row):
    return (
        f"<|im_start|>system\n{row['system']}<|im_end|>\n"    # AIへの指示部分
        f"<|im_start|>user\n{row['user']}<|im_end|>\n"        # 判定させたい会話データ
        f"<|im_start|>assistant\n{row['assistant']}<|im_end|>" # 【重要】AIに学ばせたい「模範解答（理由付き）」
    )

# データフレーム(df)の全行に対して上記の関数を実行し、合体したテキストを「text」という新しい列に保存する
df["text"] = df.apply(format_chat, axis=1)

# Hugging Faceのツールが読み込める専用のDataset形式に変換する（学習を高速化するため）
dataset = Dataset.from_pandas(df[["text"]])

# ==========================================
# 4. DGX用・高バッチサイズの学習設定
# ==========================================
print("再学習（治療）を開始します...")

# AIに「どのように学習を進めるか」のルールを設定する
training_args = SFTConfig(
    output_dir="./qwen_healed_lora", # 学習途中のバックアップなどを保存するフォルダ
    per_device_train_batch_size=8,   # 【重要】1回の学習で何問同時に解かせるか。DGXのメモリを活かして8〜32等に大きくし、学習を安定・高速化させる
    gradient_accumulation_steps=1,   # バッチサイズを大きくできたので、複数回分を合算して待つ必要はない（1に設定）
    learning_rate=1e-4,              # 学習の歩幅（0.0001）。急ぎすぎず、かつ枝刈りのショックから早く回復させるための絶妙な数値
    num_train_epochs=3,              # 【重要】用意した全データを「何周」勉強させるか。3周させてしっかり脳に定着させる
    logging_steps=10,                # 10ステップ（10回学習）ごとに、画面に学習の進み具合（損失スコアなど）を表示する
    optim="adamw_torch",             # AIの重みを最適化する最新の計算アルゴリズム。DGXなのでメモリ節約版ではなく標準の最高速版を使う
    bf16=True,                       # DGX（A100等）特有の超高速・高効率な演算モード（bfloat16）をオンにする
    save_strategy="epoch",           # 1周（1エポック）学習が終わるごとに、その時点での賢さをバックアップ保存する
    dataset_text_field="text",       # データセットの中で、AIに読ませるテキストが入っている列の名前（"text"列）
    max_length=4096              # 【重要】一度に読み込める最大文字数。20往復のような長い会話フローも最後まで読めるように2048に拡大
)

# 実際の学習を管理する「トレーナー」を呼び出し、モデル、データ、ルールをセットする
trainer = SFTTrainer(
    model=model,                     # バイパス回路を取り付けたAIモデル
    train_dataset=dataset,           # 読み込んだ学習データ
    args=training_args,              # 先ほど決めた学習ルール
    #peft_config=peft_config          # バイパス回路（LoRA）の設定
)

# 【実行】実際にAIの再学習（ファインチューニング）をスタートさせる！
trainer.train()

# ==========================================
# 5. モデルの保存
# ==========================================
# 全ての学習が完了したら、新しく賢くなった「バイパス回路（LoRAアダプタ）」の部分だけを保存する
# ※次回推論時は、枝刈りモデルとこのアダプタを合体させて使う
trainer.model.save_pretrained("./qwen_healed_lora_weights")

print("最高品質での治療が完了し、LoRAアダプタを保存しました！")