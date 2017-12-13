import nltk
from sklearn.base import BaseEstimator


class EmoticonsFeature(BaseEstimator):

    def __init__(self, positive_emoticons_dict, negative_emoticons_dict):
        self.positive_emoticons_dict = positive_emoticons_dict
        self.negative_emoticons_dict = negative_emoticons_dict
        pass

    def get_feature_names(self):
        return 'brown_cls'

    def create_emoticons_feature(self, entities):
        features = []
        for entity in entities:
            adv_num = 0
            feature = [0] * 2
            for adv in self.positive_emoticons_dict:
                if adv in entity:
                    feature[0] = 1
            for adv in self.negative_emoticons_dict:
                if adv in entity:
                    feature[1] = 1
            features.append(feature)
        return features

    def fit(self, documents, y=None):
        return self

    def transform(self, entities):
        return self.create_emoticons_feature(entities)

    def fit_transform(self, entities, y=None):
        return self.create_emoticons_feature(entities)