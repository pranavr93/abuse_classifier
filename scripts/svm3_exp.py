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
import string
from collections import Counter
from nltk import ngrams
from ast import literal_eval as make_tuple
import re
import numpy as np
import scipy
from numpy import array


def add_feature(feature, x):
    caps_np = array(feature).reshape(x.shape[0], 1)
    new_column = scipy.sparse.csr.csr_matrix(caps_np)
    x = scipy.sparse.hstack([new_column, x])
    return x


# pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.75)),
#                       ('clf', LinearSVC(random_state=0, max_iter=4000, C=0.05, class_weight='balanced'))
# ])

# read the data into pandas data frame
df_train = pd.read_csv('../data/sub_train.tsv', sep='\t', header=0)
df_test = pd.read_csv('../data/sub_test.tsv', sep='\t', header=0)

# first create the vectorizer
bigram_vectorizer = CountVectorizer(ngram_range=(1,2), max_df=0.75, stop_words='english')
x_train = bigram_vectorizer.fit_transform(df_train.comment)

# first create the vectorizer
x_test = bigram_vectorizer.transform(df_test.comment)

size = x_train.shape[0]
size_test = x_test.shape[0]

print('getting caps features ready ...')
# get the new feature

reg = re.compile(r'[a-zA-Z][\*]+[a-zA-Z]')
star_list = []
caps_list = []
leng = []
for i in range(len(df_train)):
    line = df_train.comment[i]
    num = len(filter(lambda x: x in string.uppercase, line))
    den = float(0.1+len(re.findall('[a-zA-Z]', line)))
    ratio = num/den
    # compute caps feature
    caps_list.append(ratio)

    # compute length feature
    leng.append(len(df_train.comment[i]))

    # number of stars with letters on either side
    stars = int(bool(reg.search(df_train.comment[i])))
    star_list.append(stars)

star_list_test = []
caps_list_test = []
leng_test = []

for i in range(len(df_test)):
    line = df_test.comment[i]
    num = len(filter(lambda x: x in string.uppercase, line))
    den = float(0.1+len(re.findall('[a-zA-Z]', line)))
    ratio = num/den
    # compute caps feature
    caps_list_test.append(ratio)

    # compute length feature
    leng_test.append(len(df_test.comment[i]))

    # number of stars with letters on either side
    stars = int(bool(reg.search(df_test.comment[i])))
    star_list_test.append(stars)

print('caps features done...')
# create a np array from it

x_train = add_feature(caps_list, x_train)
x_train = add_feature(leng, x_train)
x_train = add_feature(star_list, x_train)

x_test = add_feature(caps_list_test, x_test)
x_test = add_feature(leng_test, x_test)
x_test = add_feature(star_list_test, x_test)


print('caps features added to training data')
print('training model...')
linearsvc = LinearSVC(random_state=0, max_iter=4000, C=0.05, class_weight='balanced')
model = linearsvc.fit(x_train, df_train.label)

print('training complete')
######## Training complete ########

predicted = model.predict(x_train)
# metrics on training data
print('accuracy : {0}'.format(accuracy_score(df_train.label, predicted)))
print('precision : {0}'.format(precision_score(df_train.label, predicted)))
print('recall : {0}'.format(recall_score(df_train.label, predicted)))
print('f1 score : {0}'.format(f1_score(df_train.label, predicted)))


# writing results to a file
print('\n\ndetails on the train set:')
predicted = model.predict(x_test)
print('accuracy : {0}'.format(accuracy_score(df_test.label, predicted)))
print('precision : {0}'.format(precision_score(df_test.label, predicted)))
print('recall : {0}'.format(recall_score(df_test.label, predicted)))
print('f1 score : {0}'.format(f1_score(df_test.label, predicted)))

index = 0
print('\n\ndetails on the test set:')
f_out = open('../output/split_1.csv', 'w')
f_out.write("Id,Category\n")
f_bad = open('../data/bad_words.txt', 'r')
bad_ugrams = f_bad.read().splitlines()


print('Writing to file... \n')
import re
false_positive = open('../analysis/split_2_fp.txt','w')
false_negative = open('../analysis/split_2_fn.txt','w')
pattern = re.compile("[^\w']")
for item in predicted:
    temp_item = item
    if any(word.lower() in str(df_test.comment[index]).lower() for word in bad_ugrams):
        temp_item = 1

    if temp_item == 0 and df_test.label[index] == 1:
        false_negative.write('{0} Label: {1}\n{2}\n\n\n'.format(index, df_test.label[index], df_test.comment[index]))

    if temp_item == 1 and df_test.label[index] == 0:
        false_positive.write('{0} Label: {1}\n{2}\n\n\n'.format(index, df_test.label[index], df_test.comment[index]))

    f_out.write('{0},{1}\n'.format(index, temp_item))
    index += 1
