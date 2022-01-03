# -*- coding:utf-8 -*-
"""
Title
    - Practice on spam-ham classification 
Description
    - Practice on spam-ham classification that is representative in text classification task.
"""

import sys
import csv
import nltk 
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Model Creation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def preprocessing(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    # remove stopwords
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]

    # remove words less than three letter
    tokens = [word for word in tokens if len(word)>=3]

    # lower capitalization
    tokens = [word.lower() for word in tokens]

    # lemmatize
    lmtzr = WordNetLemmatizer()

    tokens = [lmtzr.lemmatize(word) for word in tokens]

    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def get_model_result(X_train, X_test, y_train, y_test):

    # TF-IDF vectorizer
    vectorizer2 = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english', strip_accents='unicode', norm='l2')
    X_train = vectorizer2.fit_transform(X_train)
    X_test = vectorizer2.transform(X_test)

    # Naive Bayes
    clf_NB = MultinomialNB().fit(X_train, y_train)
    y_predicted_NB = clf_NB.predict(X_test)

    # Decision tree
    
    clf_DT = tree.DecisionTreeClassifier().fit(X_train.toarray(), y_train)
    y_predicted_DT = clf_DT.predict(X_test.toarray())

    # Stochastic gradient descent
    
    # clf_SGD = SGDClassifier(alpha=.0001).fit(X_train,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             y_train)
    # y_predicted_SGD = clf_SGD.predict(clf_SGD)
    y_predicted_SGD = ''

    # Support Vector Machines
    
    clf_SVM = LinearSVC().fit(X_train, y_train)
    y_predicted_SVM = clf_SVM.predict(clf_SVM)

    # The Random forest algortihm
    
    clf_RFA = RandomForestClassifier(n_estimators=10)
    clf_RFA.fit(X_train, y_train)
    y_predicted_RFA = clf_RFA.predict(X_test)

    return y_predicted_NB, y_predicted_DT, y_predicted_SGD, y_predicted_SVM, y_predicted_RFA

def predicted_score(X_train, X_test, y_train, y_test, predicted_list):
    print (' \n confusion_matrix NB \n')
    cm = confusion_matrix(y_test, predicted_list[0])
    print (cm)
    print ('\n Here is the classification report:')
    print (classification_report(y_test, predicted_list[0]))

    print (' \n confusion_matrix DT \n')
    cm = confusion_matrix(y_test, predicted_list[1])
    print (cm)
    print ('\n Here is the classification report:')
    print (classification_report(y_test, predicted_list[1]))


    # print (' \n confusion_matrix SGD \n')
    # cm = confusion_matrix(y_test, predicted_list[2])
    # print (cm)
    # print ('\n Here is the classification report:')
    # print (classification_report(y_test, predicted_list[2]))


    print (' \n confusion_matrix SVM \n')
    cm = confusion_matrix(y_test, predicted_list[3])
    print (cm)
    print ('\n Here is the classification report:')
    print (classification_report(y_test, predicted_list[3]))


    print (' \n confusion_matrix RFA \n')
    cm = confusion_matrix(y_test, predicted_list[4])
    print (cm)
    print ('\n Here is the classification report:')
    print (classification_report(y_test, predicted_list[4]))



def main(argv):
    
    smsdata = open('SMSSpamCollection.txt', encoding='utf-8')

    sms_data = []
    sms_labels = []
    cnt = 0
    sencsv_reader = csv.reader(smsdata, delimiter='\t')
    for line in sencsv_reader:
        sms_labels.append(line[0])
        sms_data.append(preprocessing(line[1]))

    smsdata.close()

    
    # sploit dataset as train and test
    trainset_size = int(round(len(sms_data) * 0.70))
    print('The training set size for this classifier is ' + str(trainset_size) + '\n')
    X_train = np.array([''.join(el) for el in sms_data[0:trainset_size]])
    y_train = np.array([el for el in sms_labels[0:trainset_size]])
    X_test = np.array([''.join(el) for el in sms_data[trainset_size+1:len(sms_data)]])
    y_test = np.array(([el for el in sms_labels[trainset_size+1:len(sms_labels)]]) or el in sms_labels[trainset_size+1:len(sms_labels)])
    
    

    # 성능 검증
    predicted_list = get_model_result(X_train, X_test, y_train, y_test)
    predicted_score(X_train, X_test, y_train, y_test, predicted_list)
    

if __name__ == "__main__":
    sys.exit(main(sys.argv))