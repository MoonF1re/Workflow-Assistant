from typing import List, Tuple, Optional, Any, Dict
import os
import pickle
import difflib

try:
    from sentence_transformers import SentenceTransformer

    _HAS_TRANSFORMER = True
except Exception:
    _HAS_TRANSFORMER = False

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from core.config import FUZZY_THRESHOLD


class IntentClassifier:
    def __init__(self,
                 embed_model_name: str = "cointegrated/rubert-tiny2", #all-MiniLM-L6-v2
                 use_transformer: bool = True,
                 random_state: int = 42,
                 fuzzy_threshold: float = FUZZY_THRESHOLD):  # Порог схожести (0.85 = 85%)
        self.random_state = random_state
        self.use_transformer_requested = bool(use_transformer)
        self.use_transformer = self.use_transformer_requested and _HAS_TRANSFORMER
        self.embed_model_name = embed_model_name
        self.fuzzy_threshold = fuzzy_threshold

        self._encoder: Optional[LabelEncoder] = None
        self._clf: Optional[Any] = None
        self._vectorizer: Optional[Any] = None
        self._embedder: Optional[Any] = None

        # Хранилище примеров для Fuzzy matching: список кортежей (text, intent_label)
        self._train_examples: List[Tuple[str, str]] = []

        if self.use_transformer:
            self._embedder = None

    def _ensure_embedder(self):
        if not self.use_transformer:
            return
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embed_model_name)

    def fit(self, texts: List[str], intents: List[str], verbose: bool = True) -> Dict[str, Any]:
        """
        Обучает классификатор. Также сохраняет тексты для fuzzy-поиска.
        """
        if len(texts) != len(intents):
            raise ValueError("!!! Примеры команд и их названия должны совпадать !!!")

        # 1. Сохраняем примеры для Fuzzy Matching
        self._train_examples = list(zip(texts, intents))

        # 2. Стандартное ML обучение
        self._encoder = LabelEncoder()
        y = self._encoder.fit_transform(intents)

        if self.use_transformer:
            if verbose:
                print("[IntentClassifier] Using sentence-transformers embeddings:", self.embed_model_name)
            self._ensure_embedder()
            X = self._embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

            self._clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state)
            self._clf.fit(X, y)
            self._vectorizer = None
            return {"n_samples": len(texts), "n_classes": len(self._encoder.classes_), "backend": "transformer"}
        else:
            if verbose:
                print("[IntentClassifier] Using TF-IDF + LogisticRegression fallback")
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
            X = self._vectorizer.fit_transform(texts)
            self._clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state)
            self._clf.fit(X, y)
            return {"n_samples": len(texts), "n_classes": len(self._encoder.classes_), "backend": "tfidf"}

    def _predict_fuzzy(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Ищет наиболее похожую фразу в обучающей выборке используя расстояние Левенштейна.
        Возвращает (intent, score) или None, если ничего похожего не найдено.
        """
        if not self._train_examples:
            return None

        best_ratio = 0.0
        best_intent = None

        # Проходим по всем известным примерам
        for example_text, example_intent in self._train_examples:
            # SequenceMatcher вычисляет похожесть строк (0.0 ... 1.0)
            ratio = difflib.SequenceMatcher(None, text, example_text).ratio()

            if ratio == 1.0:
                return example_intent, 1.0

            if ratio > best_ratio:
                best_ratio = ratio
                best_intent = example_intent

        if best_ratio >= self.fuzzy_threshold:
            return best_intent, best_ratio

        return None

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Возвращает вероятности для всех команд(интент).
        """
        if self._clf is None or self._encoder is None:
            raise RuntimeError("Модель не обучена. Вызовите сначала fit().")

        # Подготовка ML (векторизация всех текстов)
        if self.use_transformer:
            self._ensure_embedder()
            X = self._embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        else:
            X = self._vectorizer.transform(texts)

        ml_probs = self._clf.predict_proba(X)
        out = []

        for i, text in enumerate(texts):
            # 1. Сначала пробуем Fuzzy
            fuzzy_res = self._predict_fuzzy(text)

            mapping = {}
            if fuzzy_res:
                found_intent, score = fuzzy_res
                # Если fuzzy нашел совпадение, ставим ему максимальный балл, остальным 0
                for cls in self._encoder.classes_:
                    mapping[cls] = score if cls == found_intent else 0.0
            else:
                # 2. Иначе используем результат ML
                row = ml_probs[i]
                for idx, p in enumerate(row):
                    mapping[self._encoder.inverse_transform([idx])[0]] = float(p)

            out.append(mapping)
        return out

    def predict(self, text: str) -> Tuple[str, float]:
        # 1. Fuzzy Match
        fuzzy_res = self._predict_fuzzy(text)
        if fuzzy_res:
            return fuzzy_res  # (intent, score)

        # 2. ML Match
        return self._predict_ml_only(text)

    def _predict_ml_only(self, text: str) -> Tuple[str, float]:
        """Чистый ML прогноз без fuzzy (вспомогательный метод)"""
        if self.use_transformer:
            self._ensure_embedder()
            X = self._embedder.encode([text], show_progress_bar=False, convert_to_numpy=True)
        else:
            X = self._vectorizer.transform([text])

        probs = self._clf.predict_proba(X)[0]
        best_idx = probs.argmax()
        best_intent = self._encoder.inverse_transform([best_idx])[0]
        return best_intent, float(probs[best_idx])

    def predict_topk(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        proba = self.predict_proba([text])[0]
        items = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)
        return items[:k]

    def save(self, path: str):
        st = {
            "use_transformer": self.use_transformer,
            "use_transformer_requested": self.use_transformer_requested,
            "embed_model_name": self.embed_model_name,
            "classes_": None if self._encoder is None else self._encoder.classes_,
            "clf": self._clf,
            "vectorizer": self._vectorizer,
            "fuzzy_threshold": self.fuzzy_threshold,
            # ВАЖНО: сохраняем примеры, чтобы fuzzy работал после перезагрузки
            "train_examples": self._train_examples
        }
        with open(path, "wb") as f:
            pickle.dump(st, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            st = pickle.load(f)
        self.use_transformer = st.get("use_transformer", False)
        self.use_transformer_requested = st.get("use_transformer_requested", False)
        self.embed_model_name = st.get("embed_model_name", self.embed_model_name)
        self.fuzzy_threshold = st.get("fuzzy_threshold", 0.85)

        # Восстанавливаем примеры
        self._train_examples = st.get("train_examples", [])

        classes = st.get("classes_", None)
        if classes is not None:
            self._encoder = LabelEncoder()
            self._encoder.classes_ = classes
        else:
            self._encoder = None

        self._clf = st.get("clf", None)
        self._vectorizer = st.get("vectorizer", None)

        if self.use_transformer:
            self._ensure_embedder()

    def fit_eval(self, texts: List[str], intents: List[str], test_size: float = 0.2, random_state: int = 42):
        # Обучение с Оценкой
        from collections import Counter
        intent_counts = Counter(intents)
        min_count = min(intent_counts.values())

        if min_count >= 2:
            X_train, X_val, y_train, y_val = train_test_split(
                texts, intents, test_size=test_size, random_state=random_state, stratify=intents
            )
        else:
            print("Предупреждение: Некоторые классы имеют <2 примеров, стратификация отключена")
            X_train, X_val, y_train, y_val = train_test_split(
                texts, intents, test_size=test_size, random_state=random_state
            )

        self.fit(X_train, y_train, verbose=True)

        preds = []
        for t in X_val:
            p, _ = self.predict(t)
            preds.append(p)
        report = classification_report(y_val, preds, zero_division=0)
        return {"report": report}


if __name__ == "__main__":
    # Тест
    texts = ["поставь таймер", "включи свет", "погода москва"]
    intents = ["timer", "light", "weather"]

    clf = IntentClassifier(use_transformer=False)
    clf.fit(texts, intents)

    # 1. Идеальное совпадение (Fuzzy 1.0)
    print("Exact match:", clf.predict("поставь таймер"))

    # 2. Опечатка (Fuzzy ~0.9)
    print("Typo match:", clf.predict("поставь таймр"))

    # 3. Совсем другое (ML)
    print("ML match:", clf.predict("хочу знать какая погода в москве"))