"""Add isMystics field to puzzles.json. First puzzle of each category = True (demo)."""
import json

with open("puzzles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for cat in data["categories"]:
    for i, puzzle in enumerate(cat["puzzles"]):
        puzzle["isMystics"] = (i == 0)

with open("puzzles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

for cat in data["categories"]:
    mystics = [p["id"] for p in cat["puzzles"] if p.get("isMystics")]
    print(f"{cat['id']}: mystics = {mystics}")
