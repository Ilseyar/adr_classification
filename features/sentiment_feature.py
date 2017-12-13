import nltk
import numpy
from dill.dill import FileNotFoundError
from pandas import DataFrame
from scipy import sparse
from sklearn import preprocessing

from sklearn.base import BaseEstimator

class SentimentFeatures(BaseEstimator):

    # window_words = []

    def __init__(self):
        pass

    def get_feature_names(self):
        return 'sentiment'

    def fit(self, documents, y=None):
        return self

    def load_sentiment_dict_subj(self):
        sentiment_dict = {}
        try:
            f = open("input/sentiment/subjclueslen1-HLTEMNLP05.tff")
            for line in f:
                terms = line.split(" ")
                sentiment_dict[terms[2][terms[2].index("=") + 1:]] = terms[len(terms) - 1][
                                                                     terms[len(terms) - 1].index("=") + 1:].strip()
        except FileNotFoundError:
            print "Please download file subjclueslen1-HLTEMNLP05.tff 3.0 from " \
                  "https://github.com/kuitang/Markovian-Sentiment/blob/master/data/subjclueslen1-HLTEMNLP05.tff" \
                  "and put it to the input/sentiment folder"
        return sentiment_dict

    def load_sentiment_dict_word_net(self):
        result = {}
        try:
            f = open("input/sentiment/SentiWordNet_3.0.txt")
            for line in f:
                if not (line.startswith("#") or line.startswith(";")):
                    line_parts = line.split("\t")
                    pos_score = line_parts[2]
                    neg_score = line_parts[3]
                    syn_terms_split = line_parts[4].split(" ")
                    for syn_term_split in syn_terms_split:
                        result[syn_term_split.split("#")[0]] = {
                            'pos_score' : float(pos_score),
                            'neg_score' : abs(float(neg_score))
                        }
                        if syn_term_split.find('weird') != -1:
                            print syn_terms_split
        except FileNotFoundError:
            print "Please download SentiWordnet 3.0 from http://sentiwordnet.isti.cnr.it/" \
                  "and put it to the input/sentiment folder"
        return result

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
        f = open("input/sentiment/negative_words.txt")
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
        sentiment_dict_subj = self.load_sentiment_dict_subj()
        sentiment_dict_word_net = self.load_sentiment_dict_word_net()
        sentiment_bingliu_dict = self.load_bingliu_dict()
        negative_words = self.load_negated_words()
        punctuation = [',', '.', '!', '?']
        for window_word in window_words:
            window_word_parts = nltk.word_tokenize(window_word)
            feature_sent_word_net = self.create_sentiment_feature_word_net(sentiment_dict_word_net, window_word_parts, negative_words, punctuation)
            feature_sent_subj = self.create_sentiment_feature_subj(sentiment_dict_subj, window_word_parts, negative_words, punctuation)
            feature_bing_liu = self.create_sentiment_feature_subj(sentiment_bingliu_dict, window_word_parts, negative_words, punctuation)
            feature = []
            feature.extend(feature_sent_word_net)
            feature.extend(feature_sent_subj)
            feature.extend(feature_bing_liu)
            features.append(feature)
        return features