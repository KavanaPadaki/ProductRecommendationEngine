import json
import pandas as pd

df = pd.read_json("data/raw/Electronics_5.json", lines=True)
df = df[["reviewerID", "asin"]].dropna().drop_duplicates()

user_ids = df["reviewerID"].unique()
item_ids = df["asin"].unique()

user_map = {u: i for i, u in enumerate(user_ids)}
item_map = {m: i for i, m in enumerate(item_ids)}

mappings = {
    "user_map": user_map,
    "item_map": item_map
}

with open("models/mappings.json", "w") as f:
    json.dump(mappings, f)

print("Saved models/mappings.json")
