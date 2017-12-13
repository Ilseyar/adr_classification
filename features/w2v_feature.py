import nltk

from sklearn.base import BaseEstimator
from sympy.printing.tests.test_numpy import np
import re
import codecs

class W2VFeatures(BaseEstimator):

    def __init__(self, model=None, num_features=None):
        self.model = model
        self.num_features = num_features
        pass

    def get_feature_names(self):
        return 'w2v'

    def fit(self, entities, y=None):
        return self

    def set_params(self, **params):
        self.test_entities = params["test_entities"]

    def makeFeatureVec(self, words, model, num_features):
        # Function to average all of the word vectors in a given
        # paragraph
        #
        # Pre-initialize an empty numpy array (for speed)
        featureVec = np.zeros((num_features,), dtype="float32")
        #
        nwords = 0.
        #
        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed
        index2word_set = set(model.index2word)
        #
        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model[word])
        #
        # Divide the result by the number of words to get the average
        # print featureVec,nwords,words
        featureVec = np.divide(featureVec, nwords)
        return featureVec

    def getAvgFeatureVecs(self, reviews, model, num_features):
        # Given a set of reviews (each one a list of words), calculate
        # the average feature vector for each one and return a 2D numpy array
        #
        # Initialize a counter
        counter = 0.
        wout = codecs.open("PubMed-and-PMC-w2v_mesh_words.txt", "w", encoding="utf-8")

        #
        # Preallocate a 2D numpy array, for speed
        reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
        #
        # Loop through the reviews

        # clean_train_reviews = []
        # for w in words:
        #     w = re.sub("\+", " ", w)
        #     clean_train_reviews.append(w.split())

        for review in reviews:
            clean_train_reviews = []
            w = re.sub("\+", " ", review)
            clean_train_reviews = [w.split()]

            for word in clean_train_reviews:
                #
                # Print a status message every 1000th review
                if counter % 1000. == 0.:
                    print "Review %d of %d" % (counter, len(reviews))
                #
                # Call the function (defined above) that makes average feature vectors
                vec1 = self.makeFeatureVec(word, model,
                                      self.num_features)
                reviewFeatureVecs[counter] = vec1

            vec1 = [str(l) for l in vec1.tolist()]
            print vec1

            wout.write(review + "\t" + "\t".join(vec1) + "\n")

            #
            # Increment the counter
            counter = counter + 1.
        return reviewFeatureVecs

    def transform(self, entities):
        return self.create_w2v_feature(entities)

    def create_w2v_feature(self, entities):
        features = []
        for entity in entities:
            text = entity
            words = nltk.word_tokenize(text)
            feature = self.makeFeatureVec(words, self.model, self.num_features)
            feature_clear = []
            for x in feature:
                if np.math.isnan(x):
                    feature_clear.append(0.0)
                else:
                    feature_clear.append(x)
            features.append(feature_clear)
        return features

    def convert_to_list(self, features):
        result = []
        for feature in features:
            result.append(list(list(feature.toarray())[0]))
        return result

    def fit_transform(self, entities, y=None):
        return self.create_w2v_feature(entities)
