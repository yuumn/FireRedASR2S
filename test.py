

def is_chinese_char(ch: str) -> bool:
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF        # CJK Unified Ideographs
         or 0x3400 <= cp <= 0x4DBF        # CJK Extension A
         or 0x20000 <= cp <= 0x2A6DF      # CJK Extension B
         or 0xF900 <= cp <= 0xFAFF        # CJK Compatibility
         or 0x2F800 <= cp <= 0x2FA1F)     # CJK Compatibility Supplement

def classify_text(text: str) -> str:
    cn, en = 0, 0
    for ch in text:
        if is_chinese_char(ch):
            cn += 1
        elif ch.isascii() and ch.isalpha():
            en += 1
    if cn and not en:
        return "zh"
    if en and not cn:
        return "en"
    if cn > en:
        return "zh"
    return "en"

x = "你好"
print(classify_text(x))

y = 8900
print(f"{y+1:03d}")