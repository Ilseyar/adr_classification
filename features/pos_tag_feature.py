
import numpy
from scipy import sparse

import nltk
from sklearn import preprocessing
from sklearn.base import BaseEstimator
import numpy as np
from pandas import DataFrame


class PosTagFeatures(BaseEstimator):

    # window_words = []

    def __init__(self):
        pass

    def get_feature_names(self):
        return 'pos_tag'

    def fit(self, documents, y=None):
        return self

    def find_pos_tag_index(self, pos_tag, pos_dict):
        index = 0
        for pos in pos_dict:
            if pos.find(pos_tag[1]) != -1:
                return index
            index += 1
        return -1

    def transform(self, window_words, y = None):
        features = []
        # pos_dict = ["NN", "NNP", "NNS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "RB", "RBR", "RBS"]
        pos_dict = ["NN", "JJ", "VB", "RB"]
        for window_word in window_words:
            feature = [0] * len(pos_dict)
            tokens = nltk.word_tokenize(window_word)
            pos_tags = nltk.pos_tag(tokens)
            for pos_tag in pos_tags:
                index = self.find_pos_tag_index(pos_tag, pos_dict)
                if index != -1:
                    feature[index] += 1
            features.append(feature)
        # X = sparse.csr_matrix(features)
        # print X
        # print X.shape
        # X_normalized = preprocessing.normalize(features, norm='l2')
        return features