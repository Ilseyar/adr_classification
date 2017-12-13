import numpy
from pandas import DataFrame
from scipy import sparse

from sklearn.base import BaseEstimator


class SentimentScore(BaseEstimator):

    # window_words = []

    def __init__(self):
        pass

    def get_feature_names(self):
        return 'sentiment_score'

    def fit(self, documents, y=None):
        return self


    def transform(self, window_words, y=None):
        f = open("input/sentiment_score_dic_with_pmi.txt", 'r')
        word_dict = {}
        for line in f:
            line_parts = line.split("\t")
            word_dict[line_parts[0]] = float(line_parts[3])
        features = []
        i = 0
        max_len = 0
        for window_word in window_words:
            window_word_parts = window_word.split(" ")
            if len(window_word_parts) > max_len:
                max_len = len(window_word_parts)
        print "Sentiment score = " + str(max_len)
        for window_word in window_words:
            window_word_parts = window_word.split(" ")
            # feature = [0] * 45
            # i = 0
            # for word in window_word_parts:
            #     if word in word_dict:
            #         feature[i] = word_dict[word]
            #     i += 1
            feature = [0]
            max = 0
            min = 0
            for word in window_word_parts:
                if word in word_dict:
                    pmi = word_dict[word]
                    feature[0] += pmi
            #         if pmi > max:
            #             max = pmi
            #         if pmi < min:
            #             min = pmi
            # feature[1] = max
            # feature[2] = min
            # feature[3] = feature[0] / len(window_word_parts)
            features.append(feature)
        # X = sparse.csr_matrix(features)
        # print X.shape
        return features