from typing import List, Tuple, Optional, Any, Dict
import os
import pickle

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


class IntentClassifier:
    def __init__(self,
                 embed_model_name: str = "all-MiniLM-L6-v2",
                 use_transformer: bool = True,
                 random_state: int = 42):
        self.random_state = random_state
        self.use_transformer_requested = bool(use_transformer)
        # фактический флаг — True только если requested и библиотека доступна
        self.use_transformer = self.use_transformer_requested and _HAS_TRANSFORMER
        self.embed_model_name = embed_model_name

        self._encoder: Optional[LabelEncoder] = None
        self._clf: Optional[Any] = None
        self._vectorizer: Optional[Any] = None
        self._embedder: Optional[Any] = None

        if self.use_transformer:
            # ленивый загрузчик
            self._embedder = None

    def _ensure_embedder(self):
        if not self.use_transformer:
            return
        if self._embedder is None:
            # загрузка модели sentence-transformers может занять время
            self._embedder = SentenceTransformer(self.embed_model_name)

    def fit(self, texts: List[str], intents: List[str], verbose: bool = True) -> Dict[str, Any]:
        """
        Обучает классификатор на списке текстов и соответствующих intent'ов.
        Возвращает словарь с обучающей статистикой.
        """
        if len(texts) != len(intents):
            raise ValueError("!!! Примеры команд и их названия должны совпадать !!!")

        self._encoder = LabelEncoder()
        y = self._encoder.fit_transform(intents)

        # если доступен transformer и выбран — используем эмбеддинги
        if self.use_transformer:
            if verbose:
                print("[IntentClassifier] Using sentence-transformers embeddings:", self.embed_model_name)
            self._ensure_embedder()
            X = self._embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

            self._clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state)
            self._clf.fit(X, y) #Обучаем на векторах и метках
            self._vectorizer = None # TF-IDF не используется
            return {"n_samples": len(texts), "n_classes": len(self._encoder.classes_), "backend": "transformer"}
        else:
            # TF-IDF pipeline
            if verbose:
                print("[IntentClassifier] Using TF-IDF + LogisticRegression fallback")
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
            # ngram_range=(1,2) - учитывает отдельные слова и пары слов
            # max_features=20000 - ограничивает размер словаря
            X = self._vectorizer.fit_transform(texts)
            self._clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state)
            self._clf.fit(X, y)
            return {"n_samples": len(texts), "n_classes": len(self._encoder.classes_), "backend": "tfidf"}

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Для каждого текста возвращает словарь {интент: вероятность}
        """
        if self._clf is None or self._encoder is None:
            raise RuntimeError("Модель не обучена. Вызовите сначала fit().")

        if self.use_transformer:
            self._ensure_embedder()
            X = self._embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        else:
            X = self._vectorizer.transform(texts)

        probs = self._clf.predict_proba(X)
        out = []
        for row in probs:
            mapping = {}
            for idx, p in enumerate(row):
                # Преобразуем числовой индекс обратно в название интента
                mapping[self._encoder.inverse_transform([idx])[0]] = float(p)
            out.append(mapping)
        return out

    def predict(self, text: str) -> Tuple[str, float]:
        # Возвращает самый вероятный интент и его вероятность для одного текста
        proba = self.predict_proba([text])[0]
        best_intent = max(proba.items(), key=lambda kv: kv[1])
        return (best_intent[0], best_intent[1])

    def predict_topk(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        # Возвращает K самых вероятных интентов, отсортированных по убыванию вероятности
        proba = self.predict_proba([text])[0]
        items = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)
        return items[:k]

    def save(self, path: str):
        # Сохранение обученной модели на диск (pickle).
        st = {
            "use_transformer": self.use_transformer,
            "use_transformer_requested": self.use_transformer_requested,
            "embed_model_name": self.embed_model_name,
            "classes_": None if self._encoder is None else self._encoder.classes_,
            "clf": self._clf,
            "vectorizer": self._vectorizer
        }
        with open(path, "wb") as f:
            pickle.dump(st, f)

    def load(self, path: str):
        # Загружает сохраненную модель из файла
        with open(path, "rb") as f:
            st = pickle.load(f)
        self.use_transformer = st.get("use_transformer", False)
        self.use_transformer_requested = st.get("use_transformer_requested", False)
        self.embed_model_name = st.get("embed_model_name", self.embed_model_name)
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

    # ---------- Convenience: quick train + evaluate ----------
    def fit_eval(self, texts: List[str], intents: List[str], test_size: float = 0.2, random_state: int = 42):
        """
        Быстро обучить и вывести оценку
        """
        from collections import Counter
        intent_counts = Counter(intents)
        min_count = min(intent_counts.values())

        if min_count >= 2:  # Если все классы имеют минимум 2 примера
            X_train, X_val, y_train, y_val = train_test_split(
                texts, intents,
                test_size=test_size,
                random_state=random_state,
                stratify=intents
            )
        else:
            print("Предупреждение: Некоторые классы имеют <2 примеров, стратификация отключена")
            X_train, X_val, y_train, y_val = train_test_split(
                texts, intents,
                test_size=test_size,
                random_state=random_state
            )
        self.fit(X_train, y_train, verbose=True)
        preds = []
        for t in X_val:
            p, _ = self.predict(t)
            preds.append(p)
        report = classification_report(y_val, preds, zero_division=0)

        """
        precision - Из всех предсказанных значений, сколько реально были правильными
        recall - Из всех реальных значений, сколько мы правильно нашли
        f1-score - Баланс между precision и recall (2×P×R/(P+R))
        support - Сколько реальных примеров этого класса в тестовых данных
        """

        return {"report": report}



# ---------------- Demo ----------------
if __name__ == "__main__":
    # Небольшой демонстрационный датасет
    texts = [
        "поставь таймер на пять минут",
        "таймер на 5 минут",
        "через три минуты поставь таймер",
        "включи свет в комнате",
        "выключи свет",
        "сделай свет красным",
        "открой сайт example.com",
        "найди в интернете новости про погоду",
        "покажи погоду в москве",
        "создай напоминание купить хлеб завтра",
    ]
    intents = [
        "set_timer",
        "set_timer",
        "set_timer",
        "turn_on_light",
        "turn_off_light",
        "set_light_color",
        "open_website",
        "search_web",
        "weather",
        "reminder",
    ]

    # Создаём классификатор (попытается использовать sentence-transformers, если есть)
    clf = IntentClassifier(use_transformer=True)
    # meta = clf.fit_eval(texts, intents, test_size=0.2)
    # print(meta['report'])
    meta = clf.fit(texts, intents)
    print("Trained classifier meta:", meta)
    # Примеры предсказаний
    queries = [
        "поставь тигмер на 3 минуты",   # ASR-опечатка -> должен выбрать set_timer
        "зажги лампу в коридоре",       # схож с turn_on_light
        "скажи мне новости о погоде",   # weather/search
        "сделай комнату голубой"        # set_light_color
    ]
    for q in queries:
        label, prob = clf.predict(q)
        top3 = clf.predict_topk(q, k=3)
        print("\nQuery:", q)
        print("Predicted:", label, f"(p={prob:.3f})")
        print("Top3:", top3)
