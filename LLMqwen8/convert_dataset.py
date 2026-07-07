import json
import pandas as pd

df = pd.read_csv("calls.csv")

with open("train.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": row["system"]
                },
                {
                    "role": "user",
                    "content": row["user"]
                },
                {
                    "role": "assistant",
                    "content": row["assistant"]
                }
            ]
        }

        f.write(
            json.dumps(
                sample,
                ensure_ascii=False
            ) + "\n"
        )