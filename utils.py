import string

from sklearn.base import TransformerMixin
import numpy as np

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.naive_bayes import CategoricalNB


nltk.download('stopwords')
nltk.download('wordnet')

wordnet_lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    # removes upper cases
    text = text.lower()
    
    # removes punctuation
    for char in string.punctuation:
        text = text.replace(char, "")
    
    #lematize the words and join back into string text
    text = " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(text)])
    return text


class CleanTextTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return np.vectorize(clean_text)(X)

    def __str__(self):
        return "CleanTextTransformer()"

    def __repr__(self):
        return self .__str__()


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()
    
    def __str__(self):
        return "DenseTransformer()"
    
    def __repr__(self):
        return self.__str__()
    

class CategoricalBatchNB(TransformerMixin):
    def __init__(self, batch_size, classes, *args, **kwargs):
        self._batch_size = batch_size
        self._classes = classes
        self._args = args
        self._kwargs = kwargs
        self._model = CategoricalNB(*args, **kwargs)

    def fit(self, x, y, **fit_params):
        batch_size = self._batch_size
        self._model = CategoricalNB(*self._args, **self._kwargs)

        for index in range(batch_size, x.shape[0] + batch_size, batch_size):
            self._model.partial_fit(
                x[index - batch_size:index, :].toarray(),
                y[index - batch_size:index],
                classes=self._classes
            )
        return self

    @staticmethod
    def transform(x, y=None, **fit_params):
        return x

    def predict(self, x):
        batch_size = self._batch_size
        predictions = []
        for index in range(batch_size, x.shape[0] + batch_size, batch_size):
            predictions.extend(
                self._model.predict(
                    x[index - batch_size:index, :].toarray()
                ).tolist()
            )
        return np.array(predictions).ravel()

    def score(self, x, y):
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)

    def __str__(self):
        return "CategoricalBatchNB()"

    def __repr__(self):
        return self.__str__()

    
    