import os
import re
import string

import snowballstemmer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder


class TextCleaner(BaseEstimator, TransformerMixin):
    def remove_mentions(self, text):
        return re.sub(r"@\s\w+", "", text)

    def remove_hashtags(self, text):
        return re.sub(r"#\s\w+", "", text)

    def remove_urls(self, text):
        return re.sub(r"http\S+", "", text)

    def only_characters(self, text):
        return re.sub("[^a-zA-Z\s]", "", text)

    def remove_extra_spaces(self, text):
        text = re.sub("\s+", " ", text)
        text = text.lstrip()
        return text.rstrip()

    def to_lower(self, text):
        return text.lower()

    def fix_words(self, text):
        text = re.sub(r"\brt\b", " ", text)
        text = re.sub(r"\bthx\b", "thanks", text)
        text = re.sub(r"\bu\b", "you", text)
        text = re.sub(r"\bhrs\b", "hours", text)
        text = re.sub(r"\baa\b", "a", text)
        text = re.sub(r"\bflightr\b", "flight", text)
        text = re.sub(r"\bur\b", "your", text)
        text = re.sub(r"\bhr\b", "hour", text)
        text = re.sub(r"\bthru\b", "through", text)
        text = re.sub(r"\br\b", "are", text)
        text = re.sub(r"\bppl\b", "people", text)
        text = re.sub(r"\btix\b", "fix", text)
        text = re.sub(r"\bplz\b", "please", text)
        text = re.sub(r"\bflightd\b", "flighted", text)
        text = re.sub(r"\btmrw\b", "tomorrow", text)
        text = re.sub(r"\bthx\b", "thanks", text)
        text = re.sub(r"\bpls\b", "please", text)
        text = re.sub(r"\bfyi\b", "for your information", text)

        text = re.sub(r"\bheyyyy\b", "hey", text)
        text = re.sub(r"\bguyyyys\b", "guys", text)
        text = re.sub(r"\byall\b", "you all", text)
        text = re.sub(r"\basap\b", "as soon as possible", text)
        text = re.sub(r"\bbtw\b", "by the way", text)
        text = re.sub(r"\bdm\b", "direct message", text)
        text = re.sub(r"\bcudtomers\b", "customers", text)
        text = re.sub(r"\bwtf\b", "what the fuck", text)
        text = re.sub(r"\biphone\b", "phone", text)
        text = re.sub(r"\bmins\b", "minutes", text)
        text = re.sub(r"\btv\b", "television", text)
        text = re.sub(r"\bokay\b", "ok", text)
        text = re.sub(r"\bfeb\b", "february", text)
        text = re.sub(r"\byr\b", "year", text)
        text = re.sub(r"\bshes\b", "she is", text)
        text = re.sub(r"\bnope\b", "no", text)
        text = re.sub(r"\bhes\b", "he is", text)
        text = re.sub(r"\btill\b", "until", text)
        text = re.sub(r"\bomg\b", "oh my god", text)
        text = re.sub(r"\btho\b", "though", text)
        text = re.sub(r"\bnothappy\b", "not happy", text)
        return re.sub(r"\bthankyou\b", "thank you", text)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        clean_X = (
            X.apply(self.remove_urls)
            .apply(self.only_characters)
            .apply(self.remove_extra_spaces)
            .apply(self.to_lower)
            .apply(self.fix_words)
        )
        return clean_X


def CleanTwitter(train, label="sentiment", text="text"):
    def tokenize(s):
        tokens = re_tok.sub(r" \1 ", s).split()
        return stemmer.stemWords(tokens)

    le = LabelEncoder()
    ct = TextCleaner()

    train["target"] = le.fit_transform(train[label])
    train["clean_text"] = ct.transform(train[text])
    re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")
    stemmer = snowballstemmer.EnglishStemmer()
    X_train, X_test, y_train, y_test = train_test_split(
        train["clean_text"].values,
        train["target"].values,
        test_size=0.25,
        random_state=0,
    )
    vect = TfidfVectorizer(
        strip_accents="unicode",
        tokenizer=tokenize,
        ngram_range=(1, 2),
        max_df=0.75,
        min_df=3,
        sublinear_tf=True,
    )

    tfidf_train = vect.fit_transform(X_train)
    tfidf_test = vect.transform(X_test)
    return X_train, X_test, y_train, y_test, tfidf_train, tfidf_test

def get_files_from_gdrive(url: str, fname: str) -> None:
    file_id = url.split("/")[5]
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, fname, quiet=False)