import numpy
import sys
from dill.dill import FileNotFoundError
from gensim.models import word2vec
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn import metrics

from features.adr_lexicon_feature import ADRLexiconFeature
from features.brown_cls import BrownClustersFeature
from features.costart_dict_feature import CostartDictionaryFeature
from features.drugname_exist import DrugnameExistFeature
from features.emoticons_feature import EmoticonsFeature
from features.extract_entities_feature import ExtractEntitiesFeature
from features.extract_window_words_feature import ExtractWindowWordsFeature
from features.pos_tag_feature import PosTagFeatures
from features.sentiment_feature import SentimentFeatures
from features.sentiment_for_CADEC_feature import SentimentForCADECFeature
from features.sentiment_score_feature import SentimentScore
from features.umls_semantic_type_feature import UMLSemanticTypeFeature
from features.w2v_feature import W2VFeatures

brown_clusters_file = "Path to brown clusters file"
umls_semantic_types_files = "input/umls_concept_id"
w2v_model_file = "Path to word2vec model file"
W2V_FEATURES_NUM = 200
costart_dict_file = "Path to COSTART lexicon"
positive_emoticons_file = "input/emoticons/positive_emoticons.txt"
negative_emoticons_file = "input/emoticons/negative_emoticons.txt"
drugnames_file = "input/drugnames_fda.txt"
adr_lexicon_file = "Path to ADR lexicon file"
sider_dict_file = "Path to SIDER lexicon"

def load_brown_clusters():
    f = open(brown_clusters_file)
    dict_cluss = {}
    for line in f:
        terms = line.split("\t")
        dict_cluss[terms[1]] = terms[2].strip()
    return dict_cluss

def load_umls_semantic_types():
    f = open(umls_semantic_types_files)
    dict_cluss = {}
    for line in f:
        line_parts = line.split("\t")
        dict_cluss[line_parts[0]] = line_parts[2].strip()
    return dict_cluss

def load_data(f):
    reviews = []
    for line in f:
        reviews.append(eval(line))
    return reviews

def load_dict(file_name):
    f = open(file_name)
    adv_dict = []
    for line in f:
        adv_dict.append(line.strip())
    return adv_dict

def extract_labels(reviews):
    labels = []
    for review in reviews:
        labels.append(review['label'])
    return labels

def extract_entities(reviews):
    entities = []
    for review in reviews:
        entities.append(review['entity'])
    return entities

def load_adr_dict():
    f = open(adr_lexicon_file)
    adr_dict = []
    for line in f:
        adr_dict.append(line.split("\t")[1])
    return adr_dict


def extract_features_for_cadec_corpus(entities, is_train):
    window_words = ExtractWindowWordsFeature().transform(entities)
    pos_tag_feature = numpy.array(PosTagFeatures().transform(window_words))
    sentiment_feature = numpy.array(SentimentForCADECFeature().transform(window_words))
    sentiment_score = numpy.array(SentimentScore().transform(window_words))
    costart_dictionary_feature = numpy.array(CostartDictionaryFeature(costart_dict).transform(window_words))
    drugnames_exist = numpy.array(DrugnameExistFeature(drugnames).transform(window_words))
    emoticons_feature = numpy.array(EmoticonsFeature(positive_emoticons_dict, negative_emoticons_dict).transform(window_words))

    entities_text = ExtractEntitiesFeature().transform(entities)
    w2v_feature = numpy.array(W2VFeatures(model=w2v_model, num_features=W2V_FEATURES_NUM).transform(entities_text))
    brown_cls_feature = numpy.array(BrownClustersFeature(brown_clusters).transform(entities_text))
    umls_semantic_type_feature = numpy.array(UMLSemanticTypeFeature(umls_semantic_types).transform(entities_text))


    if is_train:
        X = vectorizer.fit_transform(window_words)
    else:
        X = vectorizer.transform(window_words)
    X = X.toarray()

    features = numpy.concatenate((X, pos_tag_feature), axis=1)
    features = numpy.concatenate((features, sentiment_feature), axis=1)
    features = numpy.concatenate((features, sentiment_score), axis=1)
    features = numpy.concatenate((features, brown_cls_feature), axis=1)
    features = numpy.concatenate((features, umls_semantic_type_feature), axis=1)
    features = numpy.concatenate((features, w2v_feature), axis=1)
    features = numpy.concatenate((features, costart_dictionary_feature), axis=1)
    features = numpy.concatenate((features, drugnames_exist), axis=1)
    features = numpy.concatenate((features, emoticons_feature), axis=1)

    return features

def classifier_for_cadec_corpus():
    svc = LinearSVC(class_weight='auto', penalty='l2')
    f_measure = []
    entities = []
    predicted = []
    right = []
    train_data = Bunch()
    test_data = Bunch()
    for i in range(1, 6):
        print i
        f_train = open("input/cadec_corpus/" + str(i) + "/train.txt")
        f_test = open("input/cadec_corpus/" + str(i) + "/test.txt")
        train_data.reviews = load_data(f_train)
        test_data.reviews = load_data(f_test)
        train_data.labels = extract_labels(train_data.reviews)
        test_data.labels = extract_labels(test_data.reviews)
        train_data.entities = extract_entities(train_data.reviews)
        features_train = extract_features_for_cadec_corpus(train_data.reviews, True)
        svc.fit(numpy.array(features_train), numpy.array(train_data.labels))
        features_test = extract_features_for_cadec_corpus(test_data.reviews, False)
        predicted_block = svc.predict(numpy.array(features_test))
        predicted.extend(predicted_block)
        right.extend(test_data.labels)

        print metrics.f1_score(test_data.labels, predicted_block, average='macro')
        f_measure.append(metrics.f1_score(test_data.labels, predicted_block, average='macro'))

        entities.extend(test_data.reviews)

    print str(f_measure)
    print classification_report(right, predicted,  digits=3)
    print metrics.precision_score(right, predicted, average='macro')
    print metrics.recall_score(right, predicted, average='macro')
    print metrics.f1_score(right, predicted, average='macro')

def extract_features_for_twitter_corpus(entities, is_train):
    window_words = ExtractWindowWordsFeature().transform(entities)
    pos_tag_feature = numpy.array(PosTagFeatures().transform(window_words))
    sentiment_feature = numpy.array(SentimentForCADECFeature().transform(window_words))
    sentiment_score = numpy.array(SentimentScore().transform(window_words))

    entities_text = ExtractEntitiesFeature().transform(entities)
    w2v_feature = numpy.array(W2VFeatures(model=w2v_model, num_features=W2V_FEATURES_NUM).transform(entities_text))
    brown_cls_feature = numpy.array(BrownClustersFeature(brown_clusters).transform(entities_text))
    umls_semantic_type_feature = numpy.array(UMLSemanticTypeFeature(umls_semantic_types).transform(entities_text))
    # costart_dictionary_feature = numpy.array(CostartDictionaryFeature(costart_dict).transform(entities_text))
    # drugnames_exist = numpy.array(DrugnameExistFeature(drugnames).transform(window_words))
    # emoticons_feature = numpy.array(EmoticonsFeature(positive_emoticons_dict, negative_emoticons_dict).transform(entities_text))
    # adr_lexicon_feature = numpy.array(ADRLexiconFeature(adr_lexicon_dict).transform(entities_text))
    sider_dictionary_feature = numpy.array(CostartDictionaryFeature(sider_dict).transform(window_words))

    if is_train:
        X = vectorizer.fit_transform(window_words)
        # X = transformer.fit_transform(X)
    else:
        X = vectorizer.transform(window_words)
        # X = transformer.transform(X)
    X = X.toarray()

    # features = X
    features = numpy.concatenate((X, pos_tag_feature), axis=1)
    features = numpy.concatenate((features, sentiment_feature), axis=1)
    features = numpy.concatenate((features, sentiment_score), axis=1)
    features = numpy.concatenate((features, brown_cls_feature), axis=1)
    features = numpy.concatenate((features, umls_semantic_type_feature), axis=1)
    features = numpy.concatenate((features, w2v_feature), axis=1)
    # features = numpy.concatenate((features, adr_lexicon_feature), axis=1)
    features = numpy.concatenate((features, sider_dictionary_feature), axis=1)
    # features = numpy.concatenate((features, costart_dictionary_feature), axis=1)
    # features = numpy.concatenate((features, drugnames_exist), axis=1)
    # features = numpy.concatenate((features, emoticons_feature), axis=1)

    return features

def classifier_for_twitter_corpus():
    svc = LogisticRegression(class_weight='auto', penalty='l2')
    f_measure = []
    entities = []
    predicted = []
    right = []
    train_data = Bunch()
    test_data = Bunch()
    try:
        for i in range(1, 6):
                f_train = open("input/twitter_corpus/" + str(i) + "/train.txt")
                f_test = open("input/twitter_corpus/" + str(i) + "/test.txt")
                train_data.reviews = load_data(f_train)
                test_data.reviews = load_data(f_test)
                train_data.labels = extract_labels(train_data.reviews)
                test_data.labels = extract_labels(test_data.reviews)
                train_data.entities = extract_entities(train_data.reviews)
                features_train = extract_features_for_twitter_corpus(train_data.reviews, True)
                svc.fit(numpy.array(features_train), numpy.array(train_data.labels))
                features_test = extract_features_for_twitter_corpus(test_data.reviews, False)
                predicted_block = svc.predict(numpy.array(features_test))
                predicted.extend(predicted_block)
                right.extend(test_data.labels)

                print metrics.f1_score(test_data.labels, predicted_block, average='macro')
                f_measure.append(metrics.f1_score(test_data.labels, predicted_block, average='macro'))

                entities.extend(test_data.reviews)

        print str(f_measure)
        print classification_report(right, predicted, digits=3)
        print metrics.precision_score(right, predicted, average='macro')
        print metrics.recall_score(right, predicted, average='macro')
        print metrics.f1_score(right, predicted, average='macro')
    except FileNotFoundError:
        print "Please download Twitter corpus and put it into input/twitter_corpus folder"

if __name__ == '__main__':
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    transformer = TfidfTransformer()
    brown_clusters = load_brown_clusters()
    umls_semantic_types = load_umls_semantic_types()
    if w2v_model_file == "Path to word2vec model file":
        print "Please download vectors from https://github.com/dartrevan/ChemTextMining/tree/master/word2vec/Health_2.5mreviews.s200.w10.n5.v15.cbow.bin " \
              "add the path to the variable w2v_model_file"
        sys.exit()
    if brown_clusters == "Path to brown clusters file":
        print "Please download clusters from https://raw.githubusercontent.com/dartrevan/ChemTextMining/master/clustered_words/brown_clusters/brown_input-150/paths" \
              "add the path to the variable brown_clusters"
    if adr_lexicon_file == "Path to ADR lexicon file":
        print "Please download DIEGO Lab ADR Lexicon from http://diego.asu.edu/Publications/ADRSMReview/ADRSMReview.html" \
              "add the path to the variable adr_lexicon_files"
    if sider_dict_file == "Path to SIDER lexicon":
        print "Please download SIDER Lexicon from http://sideeffects.embl.de/" \
              "add the path to the variable sider_dict_file"
    if costart_dict_file == "Path to COSTART lexicon":
        print "Please download COSTART Lexicon from  https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/CST/" \
              "add the path to the variable costart_dict_file"
    w2v_model = word2vec.Word2Vec.load_word2vec_format(w2v_model_file, binary=True)
    costart_dict = load_dict(costart_dict_file)
    positive_emoticons_dict = load_dict(positive_emoticons_file)
    negative_emoticons_dict = load_dict(negative_emoticons_file)
    drugnames = load_dict(drugnames_file)
    adr_lexicon_dict = load_adr_dict()
    sider_dict = load_dict(sider_dict_file)
    classifier_for_twitter_corpus()
    # classifier_for_cadec_corpus()

