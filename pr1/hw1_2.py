from collections import Counter, defaultdict
import re
import sys

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

    class_map = {}
    
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
        # Retrieve the label
        class_num = docs[i][0]
        docs[i] = docs[i][1:]
        # Add label to mapping
        if class_num not in class_map:
            class_map[class_num] = [i]
        else:
            class_map[class_num].append(i)
        docs[i] = filter(lambda x: x not in stopwords, doc)
        docs[i] = map(lambda x: stemmer.stem(x), docs[i])
        docs[i] = map(lambda x: lemmatizer.lemmatize(x), docs[i])
        docs[i] = map(lambda x: p.number_to_words(x) if x.isdigit() else x, docs[i])
        docs[i] = list(docs[i])

    return docs, class_map


def build_matrix(docs):
    r""" Build sparse matrix from a list of documents,
    each of which is a list of word/terms in the document.
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        d = set(d)
        nnz += len(d)
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
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        length = len(keys)
        ptr[i+1] = ptr[i] + length
        n += length
        i += 1

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=numpy.double)
    mat.sort_indices()

    return mat, idx


# scale matrix and normalize its rows
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf.
    Returns scaling factors as dict. If copy is True,
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = numpy.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(nnz):
        val[i] *= df[ind[i]]

    return df if copy is False else mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm.
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/numpy.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum

    if copy is True:
        return mat


docs, class_map = normalize('train_med.dat')
mat1, idx = build_matrix(docs)
mat2 = csr_idf(mat1, copy=True)
mat3 = csr_l2normalize(mat2, copy=True)

def get_knn_centroids(docs, class_map, k=1, e=0):
    cols = docs.shape[1]
    per_partition = cols // k
    centroids = {}

    print(per_partition)

    if cols % k:
        per_partition += 1

    for i in range(k):
        for j in range(per_partition):
            print(j)
            for class_id, rows in class_map.items():
                centroids[class_id] = sum([
                    docs[row] for row in rows
                ])

    return centroids

centroids = get_knn_centroids(mat3, class_map)

def normalize2(doc):
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

def build_matrix2(docs, test_docs, idx):
    r""" Build sparse matrix from a list of documents,
    each of which is a list of word/terms in the document.
    """
    nnz = 0
    for d in test_docs:
        d = set(d)
        nnz += len(d)

    nrows = len(test_docs)
    ncols = docs.shape[1]

    # set up memory
    ind = numpy.zeros(nnz, dtype=numpy.int)
    val = numpy.zeros(nnz, dtype=numpy.double)
    ptr = numpy.zeros(nrows+1, dtype=numpy.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter

    # transfer values
    for d in test_docs:
        cnt = Counter(d)
        keys = list(filter(lambda x: x in idx, (k for k,_ in cnt.most_common())))
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        length = len(keys)
        ptr[i+1] = ptr[i] + length
        n += length
        i += 1

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=numpy.double)
    mat.sort_indices()

    return mat

def cos_sim(doc1, doc2):
    doc1_dot = doc1.dot(doc1.T)
    doc2_dot = doc2.dot(doc2.T)
    dp = doc1.dot(doc2.T)
    return dp / (doc1_dot * doc2_dot)

test_docs = normalize2('test.dat')
test_mat = build_matrix2(mat3, test_docs, idx)

with open('results.txt', 'w') as f:
    for i in range(test_mat.shape[0]):
        test_data = test_mat[i]
        high = 0
        guess = None
        for class_id, centroid in centroids.items():
            total = cos_sim(centroid, test_data)
            if total > high:
                high = total
                guess = class_id
    f.write(guess)
