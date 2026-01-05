import json
import gzip
import os

META_PATH = "data/raw/meta_Electronics.jsonl.gz"
MAPPINGS_PATH = "data/processed/mappings.json"
OUT_PATH = "models/item_meta.json"


def main():
    print("Loading item mappings...")
    with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    # item_map: asin -> item_idx
    # normalize to uppercase for matching
    asin_to_idx = {
        asin.upper(): idx
        for asin, idx in mappings["item_map"].items()
    }

    print(f"Known items from interactions: {len(asin_to_idx)}")

    item_meta = {}

    print("Scanning compressed metadata file...")
    with gzip.open(META_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            parent_asin = obj.get("parent_asin")
            title = obj.get("title")

            if not parent_asin or not title:
                continue

            key = parent_asin.upper()
            if key not in asin_to_idx:
                continue

            item_meta[key] = {
                "title": title.strip()
            }

    print(f"Collected titles for {len(item_meta)} items")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(item_meta, f, indent=2, ensure_ascii=False)

    print(f"Saved item metadata to {OUT_PATH}")


if __name__ == "__main__":
    main()
