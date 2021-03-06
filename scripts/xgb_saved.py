from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import *
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import *
from sklearn.model_selection import GridSearchCV
import sys
import math
import numpy as np
import xgboost
import pickle
import re
import string
import numpy as np
import scipy
from numpy import array

def add_feature(feature, x):
    caps_np = array(feature).reshape(x.shape[0], 1)
    new_column = scipy.sparse.csr.csr_matrix(caps_np)
    x = scipy.sparse.hstack([new_column, x])
    return x

# read the data into pandas data frame
df_train = pd.read_csv('../data/train.tsv', sep='\t', header=0)
df_test = pd.read_csv('../data/test.tsv', sep='\t', header=0)

#
# # START OF DUMP STUFF#
# # first create the vectorizer
# bigram_vectorizer = CountVectorizer(ngram_range=(1,2), max_df=0.75)
# x_train = bigram_vectorizer.fit_transform(df_train.comment)
#
# # first create the vectorizer
# x_test = bigram_vectorizer.transform(df_test.comment)
#
# size = x_train.shape[0]
# size_test = x_test.shape[0]
#
# print('getting features ready ...')
# # get the new feature
#
# reg = re.compile(r'[a-zA-Z][\*]+[a-zA-Z]')
# star_list = []
# caps_list = []
# leng = []
# for i in range(len(df_train)):
#     line = df_train.comment[i]
#     num = len(filter(lambda x: x in string.uppercase, line))
#     den = float(0.1+len(re.findall('[a-zA-Z]', line)))
#     ratio = num/den
#     # compute caps feature
#     caps_list.append(ratio)
#
#     # compute length feature
#     leng.append(len(df_train.comment[i]))
#
#     # number of stars with letters on either side
#     stars = int(bool(reg.search(df_train.comment[i])))
#     star_list.append(stars)
#
# star_list_test = []
# caps_list_test = []
# leng_test = []
#
# for i in range(len(df_test)):
#     line = df_test.comment[i]
#     num = len(filter(lambda x: x in string.uppercase, line))
#     den = float(0.1+len(re.findall('[a-zA-Z]', line)))
#     ratio = num/den
#     # compute caps feature
#     caps_list_test.append(ratio)
#
#     # compute length feature
#     leng_test.append(len(df_test.comment[i]))
#
#     # number of stars with letters on either side
#     stars = int(bool(reg.search(df_test.comment[i])))
#     star_list_test.append(stars)
#
# print('features done...')
# # create a np array from it
#
# x_train = add_feature(caps_list, x_train)
# x_train = add_feature(leng, x_train)
# x_train = add_feature(star_list, x_train)
#
# x_test = add_feature(caps_list_test, x_test)
# x_test = add_feature(leng_test, x_test)
# x_test = add_feature(star_list_test, x_test)

## END OF DUMPED STUFF ####

# # code for dump
# train_file = open('../data/train_dump_full', 'wb')
# pickle.dump(x_train, train_file)
#
# test_file = open('../data/test_dump_full', 'wb')
# pickle.dump(x_test, test_file)
# # END #

# code for load
print('loading dumps')
train_file = open('../data/train_dump', 'rb')
test_file = open('../data/test_dump', 'rb')
x_train = pickle.load(train_file)
x_test = pickle.load(test_file)
print('loading complete')
# END #

text_clf = pickle.load(open('../models/xgb_model_dump.sav', 'rb'))

# print('Fitting data to model')
# pipeline = xgboost.XGBClassifier(
#                                  max_depth=10,
#                                  learning_rate=0.1,
#                                  subsample=0.7,
#                                  colsample_bytree=0.5,
#                                  n_estimators=300,
#                                  silent=0,
#                                  objective='binary:logistic')
#
# text_clf = pipeline.fit(x_train, df_train.label)
#
# filename = '../models/xgb_model_dump.sav'
# pickle.dump(text_clf, open(filename, 'wb'))

######## Training complete ########

# predicted = text_clf.predict(x_train)
# # metrics on training data
# print('accuracy : {0}'.format(accuracy_score(df_train.label, predicted)))
# print('precision : {0}'.format(precision_score(df_train.label, predicted)))
# print('recall : {0}'.format(recall_score(df_train.label, predicted)))
# print('f1 score : {0}'.format(f1_score(df_train.label, predicted)))

predicted = text_clf.predict(x_test)
probabilities = text_clf.predict_proba(x_test)

# writing results to a file
index = 0
f_out = open('../output/xgb_4.csv', 'w')
f_out.write("Id,Category\n")

f_bad = open('../data/bad_words.txt', 'r')
bad_ugrams = f_bad.read().splitlines()

for item in predicted:
    temp_item = item
    if any(word.lower() in str(df_test.comment[index]).lower() for word in bad_ugrams):
        temp_item = 1

    f_out.write('{0},{1},{2}\n'.format(index, temp_item, probabilities[index]))
    index += 1
