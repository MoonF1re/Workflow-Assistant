from typing import List, Optional, Dict, Any
import pkg_resources
import re

# filler-слова, которые обычно можно убрать
_DEFAULT_FILLERS = {
    "эм", "эмм", "эммм", "ээ", "эээ", "мм", "ммм", "ну", "короче", "как бы", "то есть", "типа", "в общем",
    "ладно", "вот", "же"
}

# Регексы
_PUNCT_RE = re.compile(r"[^0-9A-Za-zА-Яа-яёЁ\s\-\_]")
_SPLIT_RE = re.compile(r"\s+")
_HYPHEN_RE = re.compile(r"[-\u2011\u2010]")

# Простые словари для чисел (используются, если convert_numbers=True)
_UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9
}
_TEENS = {
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
    "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19
}
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900
}
_MULTIPLIERS = {
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000,
    "миллион": 1000000, "миллиона": 1000000, "миллионов": 1000000
}
_NUMBER_WORDS = set(_UNITS) | set(_TEENS) | set(_TENS) | set(_HUNDREDS) | set(_MULTIPLIERS) | {"пол", "полтора", "полторы"}

_LEMMATIZER_AVAILABLE = False
try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
    _LEMMATIZER_AVAILABLE = True
except Exception:
    _morph = None
    _LEMMATIZER_AVAILABLE = False


class Normalizer:
    """
      - filler_words: set[str] — слова-паразиты
      - convert_numbers: bool — преобразование числительных в цифры.
      - lemmatize: bool — включить лемматизацию (исходная форма слова)
    """

    def __init__(self,
                 filler_words: Optional[set] = None,
                 convert_numbers: bool = True,
                 lemmatize: bool = False):
        self.filler_words = set(filler_words) if filler_words is not None else set(_DEFAULT_FILLERS)
        self.convert_numbers = bool(convert_numbers)
        self.lemmatize = bool(lemmatize and _LEMMATIZER_AVAILABLE)
        if lemmatize and not _LEMMATIZER_AVAILABLE:
            # не бросаем ошибку — просто отключаем лемматизацию
            self.lemmatize = False

    def _clean_text(self, text: str) -> str:
        #Нижний регистр, нормализует дефисы, удаляет прочие символы и пробелы.
        s = text.lower().replace("ё", "е")
        s = _HYPHEN_RE.sub("-", s)
        s = _PUNCT_RE.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in _SPLIT_RE.split(text.strip()) if t]

    def _replace_number_sequences(self, tokens: List[str]) -> (List[str], List[float]):
        #Заменяет последовательности слов-числительных на цифры
        out: List[str] = []
        nums: List[float] = []
        i = 0
        n = len(tokens)
        while i < n:
            tok = tokens[i]
            if tok in _NUMBER_WORDS:
                j = i
                seq = []
                while j < n and tokens[j] in _NUMBER_WORDS:
                    seq.append(tokens[j])
                    j += 1
                val = self._parse_number_tokens(seq)
                if val is not None:
                    # append numeric string (integer if whole)
                    if abs(val - int(val)) < 1e-9:
                        out.append(str(int(val)))
                        nums.append(int(val))
                    else:
                        out.append(str(val))
                        nums.append(val)
                    i = j
                    continue
                else:
                    # fallback: keep original sequence
                    out.extend(tokens[i:j])
                    i = j
                    continue
            else:
                out.append(tok)
                i += 1
        return out, nums

    def _parse_number_tokens(self, tokens: List[str]) -> Optional[float]:
        #Преобразования списка слов-числительных в число
        total = 0
        current = 0
        for t in tokens:
            if t in _HUNDREDS:
                current += _HUNDREDS[t]
            elif t in _TENS:
                current += _TENS[t]
            elif t in _TEENS:
                current += _TEENS[t]
            elif t in _UNITS:
                current += _UNITS[t]
            elif t in _MULTIPLIERS:
                mult = _MULTIPLIERS[t]
                if current == 0:
                    current = 1
                current *= mult
                total += current
                current = 0
            elif t == "пол":
                current += 0.5
            elif t in ("полтора", "полторы"):
                current += 1.5
            else:
                return None
        return total + current

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        #Приводит слова к нормальной форме
        if not self.lemmatize or not _LEMMATIZER_AVAILABLE:
            return tokens
        lemmas = []
        for t in tokens:
            p = _morph.parse(t)
            if p:
                lemmas.append(p[0].normal_form)
            else:
                lemmas.append(t)
        return lemmas

    def normalize(self, text: Optional[str]) -> Dict[str, Any]:
        #Весь пайплайн
        if text is None:
            return {"original": None, "normalized": "", "tokens": [], "lemmas": [], "numbers": []}

        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)

        if self.filler_words:
            tokens = [t for t in tokens if t not in self.filler_words]

        numbers: List[float] = []
        if self.convert_numbers:
            tokens, numbers = self._replace_number_sequences(tokens)

        lemmas: List[str] = []
        if self.lemmatize:
            lemmas = self._lemmatize_tokens(tokens)

        normalized = " ".join(tokens).strip()
        normalized = re.sub(r"\s+", " ", normalized)

        return {
            "original": text,
            "normalized": normalized,
            "tokens": tokens,
            "lemmas": lemmas,
            "numbers": numbers
        }


# --------------------------------
if __name__ == "__main__":
    n = Normalizer(convert_numbers=True, lemmatize=True)

    test = input("Введите текст для нормализации: ")

    r = n.normalize(test)
    print(f"ORIG: {r['original']}")
    print(f"NORM: {r['normalized']}")
    print(f"TOKS: {r['tokens']}")
    if r["lemmas"]:
        print(f"LEMMAS: {r['lemmas']}")
    if r["numbers"]:
        print(f"NUMBERS: {r['numbers']}")
