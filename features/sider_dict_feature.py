import nltk
from sklearn.base import BaseEstimator


class ADRLexiconFeature(BaseEstimator):

    def __init__(self, sider_dictionary):
        self.sider_dictionary = sider_dictionary
        pass

    def get_feature_names(self):
        return 'adr_lexicon_cls'

    def create_adr_lexicon_feature(self, entities):
        features = []
        for entity in entities:
            adv_num = 0
            for adv in self.sider_dictionary:
                if adv in entity:
                    adv_num += 1
                    break
            features.append([adv_num])
        # print len(features)
        return features

    def fit(self, documents, y=None):
        return self

    def transform(self, entities):
        return self.create_adr_lexicon_feature(entities)

    def fit_transform(self, entities, y=None):
        return self.create_adr_lexicon_feature(entities)