import nltk
from sklearn.base import BaseEstimator


class DrugnameExistFeature(BaseEstimator):

    def __init__(self, drugnames):
        self.drugnames = drugnames
        pass

    def get_feature_names(self):
        return 'drugname_exist'

    def create_costart_feature(self, entities):
        features = []
        for entity in entities:
            drug_num = 0
            for drug in self.drugnames:
                if drug in entity:
                    drug_num = 1
                    break
            features.append([drug_num])
        print len(features)
        return features

    def fit(self, documents, y=None):
        return self

    def transform(self, entities):
        return self.create_costart_feature(entities)

    def fit_transform(self, entities, y=None):
        return self.create_costart_feature(entities)