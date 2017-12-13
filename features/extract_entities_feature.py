from sklearn.base import BaseEstimator

class ExtractEntitiesFeature(BaseEstimator):

    def __init__(self):
        pass

    def get_feature_names(self):
        return 'extract_entities'

    def extract_entities(self, entities):
        entity_text = []
        for entity in entities:
            entity_text.append(entity['entity'])
        return entity_text

    def fit(self, entitites, y=None):
        return self

    def transform(self, entities):
        return self.extract_entities(entities)

    def fit_transform(self, entities, y = None):
        return self.extract_entities(entities)