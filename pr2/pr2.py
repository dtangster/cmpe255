#!/usr/bin/env python

from functools import partial
import os

import cv2
from numpy import linalg, vstack
from skimage.feature import hog
from skimage.transform import resize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import GaussianRandomProjection
from sklearn.tree import DecisionTreeClassifier


NEW_DIMENSION = 128


def color_histogram(image):
    return cv2.calcHist([image], [0], None, [256], [0, 256]).reshape((256,))


def oriented_gradients(image):
    image_resized = resize(image, (NEW_DIMENSION, NEW_DIMENSION), anti_aliasing=True)
    hog_image = hog(image_resized, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
    return hog_image


def load_images(folder, extract_func=oriented_gradients):
    histograms = []
    for filename in os.listdir(folder):
        abs_path = os.path.join(folder, filename)
        image = cv2.imread(abs_path, 0)
        histogram = extract_func(image)
        histograms.append(histogram)
    return vstack(histograms)


def load_labels(filename):
    with open(filename) as f:
        return list(
            filter(
                None,
                f.read().split('\n')
            )
        )


def write_labels(filename, classifications):
    classifications = map(lambda x: x + '\n', classifications)
    with open(filename, 'w') as f:
        f.writelines(classifications)


def no_op(data, labels=None):
    return data


def pca(data, labels=None):
    transformer = PCA(n_components=NEW_DIMENSION)
    return transformer.fit_transform(data, y=labels)


def lda(data, labels=None):
    transformer = LinearDiscriminantAnalysis(n_components=NEW_DIMENSION)
    return transformer.fit_transform(data, y=labels)


def svd(data, labels=None):
    transformer = TruncatedSVD(n_components=NEW_DIMENSION)
    return transformer.fit_transform(data, y=labels)


def random_projection(data, labels=None):
    transformer = GaussianRandomProjection(n_components=NEW_DIMENSION)
    return transformer.fit_transform(data, y=labels)


def knn(data, labels, test_data):
    neighbors = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=28)
    neighbors.fit(data, labels[:data.shape[0]])
    return neighbors.predict(test_data)


def naive_bayes(data, labels, test_data):
    nb = GaussianNB()
    nb.fit(data, labels[:data.shape[0]])
    return nb.predict(test_data)


def decision_trees(data, labels, test_data):
    dt = DecisionTreeClassifier()
    dt.fit(data, labels[:data.shape[0]])
    return dt.predict(test_data)


def ada_boost(data, labels, test_data):
    ab = AdaBoostClassifier()
    ab.fit(data, labels[:data.shape[0]])
    return ab.predict(test_data)


def classify(train_data, test_data, labels, reduce_func=svd, classifier_func=knn):
    train_data = reduce_func(train_data, labels)
    test_data = reduce_func(test_data)
    return classifier_func(train_data, labels, test_data)


if __name__ == '__main__':
    train_labels = load_labels('/scratch/cmpe255-sp19/data/pr2/traffic/train.labels')
    train_data = load_images('/scratch/cmpe255-sp19/data/pr2/traffic/train')
    test_data = load_images('/scratch/cmpe255-sp19/data/pr2/traffic/test')
    classifications = classify(
        train_data, test_data, train_labels, reduce_func=svd, classifier_func=knn
    )
    write_labels('test.txt', classifications)
