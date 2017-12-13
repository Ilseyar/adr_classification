import nltk
from sklearn.base import BaseEstimator


class CostartDictionaryFeature(BaseEstimator):

    def __init__(self, costart_dictionary):
        self.costart_dictionary = costart_dictionary
        pass

    def get_feature_names(self):
        return 'brown_cls'

    def create_costart_feature(self, entities):
        features = []
        for entity in entities:
            adv_num = 0
            for adv in self.costart_dictionary:
                if adv in entity:
                    adv_num = 1
                    break
            features.append([adv_num])
        print len(features)
        return features

    def fit(self, documents, y=None):
        return self

    def transform(self, entities):
        return self.create_costart_feature(entities)

    def fit_transform(self, entities, y=None):
        return self.create_costart_feature(entities)