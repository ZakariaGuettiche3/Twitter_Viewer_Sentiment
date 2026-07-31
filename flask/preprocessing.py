

import re
import string

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()


_STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "but"}


def lemmatize_text(text: str) -> str:
    """Lemmatize every word in a whitespace-separated string."""
    return " ".join(_lemmatizer.lemmatize(word) for word in str(text).split())




def clean_single_text(text: str) -> str:


    text = str(text).lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\n", " ", text)

    text = " ".join(word for word in text.split() if word not in _STOP_WORDS)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = lemmatize_text(text)

    return text


def encode_sentence(sentence, MAX_LEN, word2idx):
    
    ids = []

    for word in sentence.split():
        ids.append(
            word2idx.get(word, 1)
        )

    ids = ids[:MAX_LEN]

    ids += [0] * (MAX_LEN - len(ids))

    return ids
