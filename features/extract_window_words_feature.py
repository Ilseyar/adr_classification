from sklearn.base import BaseEstimator

class ExtractWindowWordsFeature(BaseEstimator):

    def __init__(self):
        pass

    def get_feature_names(self):
        return 'extract_window_words'

    def extract_window_words(self, entities):
        window_words = []
        for entity in entities:
            window_words.append(entity['text'])
        return window_words

    def fit(self, entities, y=None):
        return self

    def transform(self, entities):
        print "extract word transform"
        return self.extract_window_words(entities)

    def fit_transform(self, entities, y = None):
        print "extract word fit_transform"
        return self.extract_window_words(entities)
