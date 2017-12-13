import nltk

from sklearn.base import BaseEstimator

class BrownClustersFeature(BaseEstimator):

    def __init__(self, brown_clusters):
        self.brown_clusters = brown_clusters
        pass

    def get_feature_names(self):
        return 'brown_cls'

    def create_brown_cluster_feature(self, entities):
        features = []
        cluster_number = set()
        keys = self.brown_clusters.keys()
        for key in keys:
            cluster_number.add(self.brown_clusters[key])
        cluster_number_list = list(cluster_number)
        for entity in entities:
                words = nltk.word_tokenize(entity)
                feature = [0] * len(cluster_number)
                for word in words:
                    if word in self.brown_clusters:
                        brown_cluster_num = self.brown_clusters[word]
                        feature[cluster_number_list.index(brown_cluster_num)] = 1
                features.append(feature)
        return features

    def fit(self, documents, y=None):
        return self

    def transform(self, entities):
        return self.create_brown_cluster_feature(entities)

    def fit_transform(self, entities, y=None):
        return self.create_brown_cluster_feature(entities)