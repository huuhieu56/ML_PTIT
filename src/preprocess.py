import re
import unicodedata
from underthesea import word_tokenize

TEENCODE = {
    "k": "không", "ko": "không", "kh": "không", "khg": "không", "hok": "không",
    "đc": "được", "dc": "được", "dk": "được", "sp": "sản phẩm", "dt": "điện thoại",
    "đt": "điện thoại", "lap": "laptop", "tl": "trả lời", "rep": "trả lời", "ib": "nhắn tin",
    "ok": "tốt", "okie": "tốt", "oke": "tốt", "ship": "giao hàng", "nhah": "nhanh",
    "qúa": "quá", "wa": "quá", "tks": "cảm ơn", "thanks": "cảm ơn", "r": "rồi",
    "rùi": "rồi", "lun": "luôn", "dởm": "kém chất lượng", "fake": "hàng giả"
}

EMOJIS = {
    "❤️": " emoji_pos ", "💖": " emoji_pos ", "👍": " emoji_pos ", "⭐": " emoji_pos ",
    "😍": " emoji_pos ", "😊": " emoji_pos ", "👎": " emoji_neg ", "😡": " emoji_neg ",
    "🤬": " emoji_neg ", "💔": " emoji_neg ", "😭": " emoji_neg ", "💩": " emoji_neg "
}

def clean_text(text: str, is_vietnamese: bool = True) -> str:
    text = unicodedata.normalize("NFC", str(text))
    for emoji, token in EMOJIS.items():
        text = text.replace(emoji, token)
    text = re.sub(r"https?://\S+|www\.\S+|<.*?>", " ", text.lower())
    text = re.sub(r"([!?.,~])\1+", r"\1", text)
    if is_vietnamese:
        text = " ".join(TEENCODE.get(w, w) for w in text.split())
        text = word_tokenize(text, format="text")
    return re.sub(r"\s+", " ", text).strip()

def rating_to_sentiment(rating: int) -> int:
    return 0 if rating <= 2 else (1 if rating == 3 else 2)
