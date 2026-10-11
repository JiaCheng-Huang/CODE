import emoji


def extract_emojis(text):
    # 遍历字符串，如果是 emoji 就保留
    return "".join(c for c in text if c in emoji.EMOJI_DATA)


def get_emoji_name(emoji_char):
    name_with_colons = emoji.demojize(emoji_char)
    clean_name = name_with_colons.replace(':', '').replace('_', ' ')

    return clean_name


if __name__ == '__main__':
    content = "He sent me a 🍆 last night."
    print(extract_emojis(content))
