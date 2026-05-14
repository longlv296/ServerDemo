"""Add isJigsaw fields to puzzles.json. Set 2nd puzzle of each category as jigsaw demo."""
import json

with open("puzzles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for cat in data["categories"]:
    for i, puzzle in enumerate(cat["puzzles"]):
        # Add defaults for all
        if "isJigsaw" not in puzzle:
            puzzle["isJigsaw"] = False
        if "jigsawGrid" not in puzzle:
            puzzle["jigsawGrid"] = None
        if "jigsawAdSlots" not in puzzle:
            puzzle["jigsawAdSlots"] = None

        # Demo: 2nd puzzle of each category = jigsaw 2x2
        if i == 1:
            puzzle["isJigsaw"] = True
            puzzle["jigsawGrid"] = "2x2"
            puzzle["jigsawAdSlots"] = [1, 2, 3]  # ads on quadrants 1,2,3

with open("puzzles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

for cat in data["categories"]:
    jigsaws = [p["id"] for p in cat["puzzles"] if p.get("isJigsaw")]
    if jigsaws:
        print(f"{cat['id']}: jigsaw = {jigsaws}")
