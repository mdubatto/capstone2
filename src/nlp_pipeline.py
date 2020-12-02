 #!/usr/bin/python
 # -*- coding: utf-8 -*-
import numpy as np
import string
import unicodedata

import nltk

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
from nltk import pos_tag
from nltk import RegexpParser
import gensim
from sklearn.model_selection import train_test_split



def extract_bow_from_raw_text(text_as_string):
    """Extracts bag-of-words from a raw text string.

    Parameters
    ----------
    text (str): a text document given as a string

    Returns
    -------
    list : the list of the tokens extracted and filtered from the text
    """
    if (text_as_string == None):
        return []

    if (len(text_as_string) < 1):
        return []
    text_as_string = text_as_string.lower()
    tokens = word_tokenize(text_as_string)
    # stopwords_ = set(stopwords.words('english'))
    tokens_wo_stop = [w for w in tokens if w not in string.punctuation]
    wordnet = WordNetLemmatizer()
    tokens_lemm = list(map(wordnet.lemmatize, tokens_wo_stop))
    
    return tokens_lemm

def vectorize(bow, model, size):
    """
    Converts headlines into vectors by adding the word vectors.
    """
    corpus_vec = np.zeros((len(bow),size))
    for i, row in enumerate(bow):
        row_vec = np.zeros(size)
        for word in row:
            try:
                row_vec += model.wv[word]
            except:
                pass
        corpus_vec[i] = row_vec/len(row)
    return corpus_vec

def word_embed(bow_train, bow_test, min_count=2, size=100, seed=2):

    model = gensim.models.Word2Vec(bow_train, min_count = min_count, size = size, window = 5, seed=seed)

    vec_train = vectorize(bow_train, model, size)
    vec_test = vectorize(bow_test, model, size)
    return vec_train, vec_test



