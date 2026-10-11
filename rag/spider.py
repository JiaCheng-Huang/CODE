import os
import requests
from bs4 import BeautifulSoup
import json
import time
import re
import random

# ---------------- Configuration ----------------
FOLDER_PATH = "emoji_index"
OUTPUT_FILE = "emoji_kb.json"
BASE_URL = "https://emojipedia.org/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

EMOJI_PATTERN = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)

# 关键词规则：越具体的越靠前
CATEGORY_RULES = [
    ("animals & nature",     "Animals & Nature"),
    ("animals and nature",   "Animals & Nature"),
    ("food & drink",         "Food & Drink"),
    ("food and drink",       "Food & Drink"),
    ("travel & places",      "Travel & Places"),
    ("travel and places",    "Travel & Places"),
    ("smileys & people",     "Smileys & People"),
    ("smileys and people",   "Smileys & People"),
    ("smileys",              "Smileys & People"),
    ("smiley",               "Smileys & People"),
    ("people",               "Smileys & People"),
    ("activity",             "Activity"),
    ("activities",           "Activity"),
    ("flags",                "Flags"),
    ("flag",                 "Flags"),
    ("objects",              "Objects"),
    ("object",               "Objects"),
    ("symbols",              "Symbols"),
    ("symbol",               "Symbols"),
]


# ---------------- Helpers ----------------

def _has_scroll_mt(classes):
    """判断 class 里是否包含 scroll-mt，用于定位子分类容器"""
    if not classes:
        return False
    if isinstance(classes, str):
        return "scroll-mt" in classes
    return any("scroll-mt" in c for c in classes)


def infer_big_category(file_name, soup):
    """大类判断：文件名 → <title> → <h1>，逐级兜底"""
    candidates = [os.path.basename(file_name)]

    title = soup.find("title")
    if title:
        candidates.append(title.get_text(" ", strip=True))

    h1 = soup.find("h1")
    if h1:
        candidates.append(h1.get_text(" ", strip=True))

    for text in candidates:
        low = text.lower()
        for key, cat in CATEGORY_RULES:
            if key in low:
                return cat, text

    return "Unknown", candidates[0] if candidates else ""


# ---------------- Index Extraction ----------------

def batch_extract_index():
    if not os.path.exists(FOLDER_PATH):
        print(f"[ERROR] Directory not found: {FOLDER_PATH}")
        return []

    index_data = []
    seen_slugs = set()
    files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith('.html')]
    print(f"[INFO] Initializing index scan. Found {len(files)} source files.\n")

    for file_name in files:
        file_full_path = os.path.join(FOLDER_PATH, file_name)
        try:
            with open(file_full_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            big_category, matched_text = infer_big_category(file_name, soup)
            sections = soup.find_all('div', id=True, class_=_has_scroll_mt)

            file_new_items = 0
            skipped_no_h2 = 0

            for section in sections:
                # 没有 <h2> 的容器直接跳过（通常是广告位、侧边栏等）
                h2 = section.find('h2')
                if not h2:
                    skipped_no_h2 += 1
                    continue
                sub_category = h2.get_text(" ", strip=True)

                for a in section.find_all('a', href=True):
                    href = a['href'].strip('/')
                    if not href or '#' in href:
                        continue

                    slug = href.split('/')[-1]
                    if slug in seen_slugs:
                        continue

                    link_text = a.get_text(strip=True)
                    match = EMOJI_PATTERN.search(link_text)
                    emoji_icon = match.group(0) if match else ""
                    if not emoji_icon or not slug:
                        continue

                    index_data.append({
                        "emoji": emoji_icon,
                        "char": slug,
                        "category": big_category,
                        "subcategory": sub_category,
                    })
                    seen_slugs.add(slug)
                    file_new_items += 1

            flag = "  <<< 未识别大类" if big_category == "Unknown" else ""
            print(f"[SUCCESS] {file_name}")
            print(f"    big={big_category}  matched={matched_text!r}{flag}")
            print(f"    sections={len(sections)}  skipped_no_h2={skipped_no_h2}  new_items={file_new_items}")

        except Exception as e:
            print(f"[WARNING] Error processing {file_name}: {str(e)}")

    print(f"\n[SUMMARY] Indexing complete. Total unique emojis: {len(index_data)}\n")
    return index_data


# ---------------- Utility and Crawler ----------------

def clean_text(text):
    """Sanitizes text for vector embedding quality."""
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('\ufe0f', '')
    text = re.sub(r'A new emoji quiz game.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    return re.sub(r'\n{2,}', '\n\n', text).strip()


def crawl_emoji_knowledge():
    """Crawls detailed emoji data and saves in a structured format with ID."""
    kb_results = []
    existing_slugs = set()
    current_max_id = 0

    # 1. Resume support
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    kb_results = data
                    existing_slugs = {item['char'] for item in kb_results}
                    if kb_results:
                        current_max_id = max(item['id'] for item in kb_results)
                else:
                    print("[WARNING] Old format detected. Starting fresh.")
            print(f"[RESUME] Found {len(kb_results)} entries. Current Max ID: {current_max_id}")
        except Exception as e:
            print(f"[INFO] Error loading cache ({e}). Starting fresh dataset.")

    # 2. Index from local HTML
    emoji_index = batch_extract_index()
    if not emoji_index:
        print("[ERROR] Index is empty. Terminating.")
        return

    slug_meta = {item['char']: item for item in emoji_index}

    # 3. 回填已有条目缺失/Unknown 的 category / subcategory
    backfilled_cat = 0
    backfilled_sub = 0
    for entry in kb_results:
        meta = slug_meta.get(entry.get('char'))
        if not meta:
            continue
        if entry.get('category') in (None, "", "Unknown"):
            entry['category'] = meta['category']
            backfilled_cat += 1
        if entry.get('subcategory') in (None, "", "Unknown"):
            entry['subcategory'] = meta['subcategory']
            backfilled_sub += 1

    if backfilled_cat or backfilled_sub:
        print(f"[RESUME] Backfilled category for {backfilled_cat} entries, "
              f"subcategory for {backfilled_sub} entries.")

    print("\n=== Phase 2: Remote Semantic Acquisition ===")

    id_counter = current_max_id + 1

    for idx, item in enumerate(emoji_index, 1):
        slug = item['char']
        emoji_icon = item['emoji']
        category = item.get('category', 'Unknown')
        subcategory = item.get('subcategory', 'Unknown')

        if slug in existing_slugs:
            continue

        url = f"{BASE_URL}{slug}/"
        try:
            print(f"[{idx}/{len(emoji_index)}] Crawling: {slug} "
                  f"({category} / {subcategory})...", end=" ")

            res = requests.get(url, headers=HEADERS, timeout=12)
            if res.status_code != 200:
                print(f"[FAILED] HTTP {res.status_code}")
                continue

            soup = BeautifulSoup(res.text, 'html.parser')
            container = soup.find('div', {
                "class": re.compile(r'HtmlContent_html-content-container'),
                "data-size": "default"
            })

            if container:
                paras = [clean_text(p.get_text())
                         for p in container.find_all('p')
                         if len(p.get_text()) > 25]
                description = "\n\n".join(paras)

                if description:
                    entry = {
                        "id": id_counter,
                        "emoji": emoji_icon,
                        "char": slug,
                        "category": category,
                        "subcategory": subcategory,
                        "description": description,
                    }
                    kb_results.append(entry)
                    print(f"[DONE] ID: {id_counter} | {emoji_icon} | {category}")
                    id_counter += 1
                else:
                    print("[SKIP] No valid description.")
            else:
                print("[SKIP] Container missing.")

            time.sleep(random.uniform(1.2, 2.5))

            if id_counter % 15 == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(kb_results, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"[ERROR] {slug} crashed: {str(e)}")
            time.sleep(5)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(kb_results, f, ensure_ascii=False, indent=4)

    # 统计一下最终 Unknown 数量
    unknown_cat = sum(1 for e in kb_results if e.get('category') in (None, "", "Unknown"))
    unknown_sub = sum(1 for e in kb_results if e.get('subcategory') in (None, "", "Unknown"))

    print(f"\n[COMPLETE] Knowledge base finalized with {len(kb_results)} entries.")
    print(f"[STAT] Unknown category: {unknown_cat} | Unknown subcategory: {unknown_sub}")


if __name__ == "__main__":
    crawl_emoji_knowledge()