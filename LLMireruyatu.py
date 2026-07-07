from huggingface_hub import snapshot_download

# プロジェクトディレクトリを基準として、"./llm/snapshot/"にモデルファイルが保存される。

snapshot_download(repo_id = "Qwen/Qwen3.5-4B", local_dir="/home/tokusagi/tokushusagi/qwen38B")
