from sklearn.feature_extraction.text import *
from nltk.stem import PorterStemmer
from nltk.sentiment.util import mark_negation
from nltk.corpus import opinion_lexicon
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score

import util
import re
import time


# import nltk
# nltk.download('opinion_lexicon')


def stem(message):
    ps = PorterStemmer()

    sentence = ""
    for word in message.split():
        sentence += " " + ps.stem(word)

    return sentence


def only_opinion_lexicon(message):
    sentence = ""
    ol = set(opinion_lexicon.words())
    for word in message.split():
        if word in ol or word[:-4] in ol:
            sentence += " " + word
    return sentence


def clean(messages, stemming=True, negation=True, ol=True):
    start = time.time()
    for msg_idx in range(len(messages)):
        # basic pre-processing
        messages[msg_idx] = messages[msg_idx].lower()
        messages[msg_idx] = re.sub("<.*/><.*/>", " ", messages[msg_idx])
        messages[msg_idx] = re.sub("'", "", messages[msg_idx])
        messages[msg_idx] = re.sub("\.", " . ", messages[msg_idx])
        # negation
        if negation:
            messages[msg_idx] = ' '.join(mark_negation(messages[msg_idx].split()))

        messages[msg_idx] = re.sub("[^_0-9A-Za-z]", " ", messages[msg_idx])
        messages[msg_idx] = re.sub("\s+", " ", messages[msg_idx])

        # stemming
        if stemming:
            messages[msg_idx] = stem(messages[msg_idx])

        # filtering based off opinion lexicon
        if ol:
            messages[msg_idx] = only_opinion_lexicon(messages[msg_idx])
    print("Cleaning Time: ", time.time() - start, " seconds")
    return messages


def main():
    train_messages, train_labels = util.load_review_dataset('reviews_train.csv')
    test_messages, test_labels = util.load_review_dataset('reviews_test.csv')

    train_messages = clean(train_messages, stemming=True, ol=True)
    test_messages = clean(test_messages, stemming=True, ol=True)

    cv = CountVectorizer(stop_words='english', min_df=0)
    cv_train_matrix = cv.fit_transform(train_messages)
    cv_test_matrix = cv.transform(test_messages)

    tv = TfidfVectorizer(stop_words='english', min_df=25)
    tv_train_matrix = tv.fit_transform(train_messages)
    tv_test_matrix = tv.transform(test_messages)

    print("Vocabulary size: ", len(cv.vocabulary_))

    cv_model = SVC(kernel='linear', C=1, gamma=1)
    cv_scores = cross_val_score(cv_model, cv_train_matrix, train_labels, scoring='accuracy', cv=5)
    cv_model.fit(cv_train_matrix, train_labels)
    cv_predict = cv_model.predict(cv_test_matrix)

    tv_model = SVC(kernel='linear', C=1, gamma=1)
    tv_scores = cross_val_score(tv_model, tv_train_matrix, train_labels, scoring='accuracy', cv=5)
    tv_model.fit(tv_train_matrix, train_labels)
    tv_predict = tv_model.predict(tv_test_matrix)

    print("BoW Acc: ", metrics.accuracy_score(test_labels, cv_predict))
    print("BoW Cross Validated: ", cv_scores.mean())
    print("TF-IDF Acc: ", metrics.accuracy_score(test_labels, tv_predict))
    print("TF-IDF Cross Validated: ", tv_scores.mean())

    # for i in range(len(test_labels)):
    #     if test_labels[i] != cv_predict[i]:
    #         print("index: ", i, " BoW fail: ", test_messages[i], " label: ", test_labels[i])
    #     if test_labels[i] != tv_predict[i]:
    #         print("index: ", i, " TFIDF fail: ", test_messages[i], " label: ", test_labels[i])


if __name__ == "__main__":
    main()
