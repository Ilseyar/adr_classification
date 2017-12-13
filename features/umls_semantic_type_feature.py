import nltk
from pandas import DataFrame

from sklearn.base import BaseEstimator

class UMLSemanticTypeFeature(BaseEstimator):

    def __init__(self, semantic_types):
        # self.entities = train_entities
        self.semantic_types = semantic_types
        pass

    def get_feature_names(self):
        return 'umls_semant_type'


    # def load_semantic_types(self):
    #     f = open("input/umls/umls_concept_id")
    #     dict_cluss = {}
    #     for line in f:
    #         line_parts = line.split("\t")
    #         dict_cluss[line_parts[0]] = line_parts[2].strip()
    #     return dict_cluss

    def create_semantic_type_feature(self, entities):
        features = []
        semantic_types_set = set()
        keys = self.semantic_types.keys()
        for key in keys:
            semantic_types_set.add(self.semantic_types[key])
        cluster_number_list = list(semantic_types_set)
        for entity in entities:
                feature = [0] * len(self.semantic_types)
                if entity in self.semantic_types:
                    semantic_type =  self.semantic_types[entity]
                    feature[cluster_number_list.index(semantic_type)] = 1
                features.append(feature)
        return features

    def fit(self, documents, y=None):
        return self

    def transform(self, entities):
        return self.create_semantic_type_feature(entities)

    def fit_transform(self, entities, y=None):
        return self.create_semantic_type_feature(entities)
