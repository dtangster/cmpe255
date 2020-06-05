#!/usr/bin/env python

from collections import Counter
import re

import contractions
import inflect
import nltk
import numpy
import scipy

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

from scipy.sparse import csr_matrix


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = set(stopwords.words('english'))
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
p = inflect.engine()

def normalize(doc):
    with open(doc, encoding='utf8') as f:
        docs = f.readlines()

    # Remove punctuation
    docs = map(lambda x: re.sub(r'[^\w\s]', '', x), docs)
    # Lower-case all words
    docs = map(str.lower, docs)
    # Change words like "aren't" -> "are not"
    docs = map(contractions.fix, docs)
    # Transform each document into an array of words
    docs = map(nltk.word_tokenize, docs)
    docs = list(docs)
    # Remove stop words
    for i, doc in enumerate(docs):
        docs[i] = filter(lambda x: x not in stopwords, doc)
        docs[i] = map(lambda x: stemmer.stem(x), docs[i])
        docs[i] = map(lambda x: lemmatizer.lemmatize(x), docs[i])
        docs[i] = map(lambda x: p.number_to_words(x) if x.isdigit() else x, docs[i])
        docs[i] = list(docs[i])
    
    return docs


def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = numpy.zeros(nnz, dtype=numpy.int)
    val = numpy.zeros(nnz, dtype=numpy.double)
    ptr = numpy.zeros(nrows+1, dtype=numpy.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=numpy.double)
    mat.sort_indices()
    
    return mat


if __name__ == '__main__':
    docs = normalize('sample.dat')
    print(len(docs[0]))
    matrix = build_matrix(docs)
    print(matrix[0])
