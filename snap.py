from huggingface_hub import snapshot_download

# プロジェクトディレクトリを基準として、"./llm/snapshot/"にモデルファイルが保存される。

snapshot_download(repo_id = "google/gemma-4-E4B-it", local_dir="./tokushusagi")
