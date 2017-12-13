import nltk
import numpy
from dill.dill import FileNotFoundError
from pandas import DataFrame
from scipy import sparse
from sklearn import preprocessing

from sklearn.base import BaseEstimator

class SentimentForCADECFeature(BaseEstimator):

    # window_words = []

    def __init__(self):
        pass

    def get_feature_names(self):
        return 'sentiment'

    def fit(self, documents, y=None):
        return self

    def create_sentiment_feature_word_net(self, sentiment_dict, window_word, negative_words, punctuation):
        feature = [0] * 8
        zero_pmi_score = 0
        total_score = 0
        max_score = 0
        last_score = 0
        zero_pmi_score_neg = 0
        total_score_neg = 0
        max_score_neg = 0
        last_score_neg = 0
        is_context_negated = False
        for word in window_word:
            if word in negative_words:
                is_context_negated = True
            elif word in punctuation:
                is_context_negated = False
            if word in sentiment_dict:
                sentiment = sentiment_dict[word]
                pos = float(sentiment['pos_score'])
                neg = float(sentiment['neg_score'])
                score = pos - neg
                if is_context_negated:
                    if score != 0:
                        zero_pmi_score_neg += 1
                    total_score_neg += score
                    if score != 0 and (score > max_score_neg or (abs(score) > max_score_neg and max_score_neg == 0)):
                        max_score_neg = score
                    last_score_neg = score
                else:
                    if score != 0:
                        zero_pmi_score += 1
                        if score > max_score or ( abs(score) > max_score and max_score == 0):
                            max_score = score
                        last_score = score
                    total_score += score
        feature[0] = zero_pmi_score
        feature[1] = total_score
        feature[2] = max_score
        feature[3] = last_score
        feature[4] = zero_pmi_score_neg
        feature[5] = total_score_neg
        feature[6] = max_score_neg
        feature[7] = last_score_neg
        return feature


    def load_negated_words(self):
        f = open("input/negative_words.txt")
        negative_words_dict = []
        for line in f:
            negative_words_dict.append(line.strip())
        return negative_words_dict

    def create_sentiment_feature_subj(self, sentiment_dict, window_word, negative_words, punctuation):
        feature = [0] * 4
        positive_affirmative = 0
        negative_affirmative = 0
        positive_negated = 0
        negative_negated = 0
        is_negative_context = False
        for word in window_word:
            if word in negative_words:
                is_negative_context = True
            elif word in punctuation:
                is_negative_context = False
            if word in sentiment_dict:
                sentiment = sentiment_dict[word].strip()
                if sentiment == "negative":
                    if is_negative_context:
                        negative_negated += 1
                    else:
                        negative_affirmative += 1
                elif sentiment == "positive":
                    if is_negative_context:
                        positive_negated += 1
                    else:
                        positive_affirmative += 1
        feature[0] = positive_affirmative
        feature[1] = negative_affirmative
        feature[2] = positive_negated
        feature[3] = negative_negated
        return feature

    def load_bingliu_dict(self):
        bingliu_dict = {}
        try:
            f_negative = open("input/sentiment/bingliunegs.txt")
            for line in f_negative:
                bingliu_dict[line.strip()] = "negative"
            f_positive = open("input/sentiment/bingliuposs.txt")
            for line in f_positive:
                bingliu_dict[line.strip()] = "positive"
        except FileNotFoundError:
            print "Please download Opinion Lexicon from https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon" \
                  "and put it to the input/sentiment folder"
        return bingliu_dict

    def transform(self, window_words, y=None):
        features = []
        sentiment_bingliu_dict = self.load_bingliu_dict()
        for window_word in window_words:
            window_word_parts = nltk.word_tokenize(window_word)
            feature = [0] * 2
            for window_word in window_word_parts:
                if window_word in sentiment_bingliu_dict:
                    sentiment = sentiment_bingliu_dict[window_word]
                    if sentiment == 'negative':
                        feature[0] = 1
                    elif sentiment == 'positive':
                        feature[1] = 1
            features.append(feature)
        return features