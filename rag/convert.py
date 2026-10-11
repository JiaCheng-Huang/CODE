import json


def to_flashrag_format():
    with open("emoji_kb.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("./emoji_corpus.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            doc = {
                "id": str(item["id"]),
                "contents": "name" + ":" + item["char"] + "," + "emoji" + ":" + item[
                    "emoji"] + "," + "description" + ":" + item["description"]
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print("Done: emoji_corpus.jsonl created.")


to_flashrag_format()
