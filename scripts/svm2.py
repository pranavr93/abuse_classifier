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


# pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.75)),
#                       ('clf', LinearSVC(random_state=0, max_iter=4000, C=0.05, class_weight='balanced'))
# ])

# read the data into pandas data frame
df_train = pd.read_csv('../data/train.tsv', sep='\t', header=0)
df_test = pd.read_csv('../data/test.tsv', sep='\t', header=0)

# first create the vectorizer
bigram_vectorizer = CountVectorizer(ngram_range=(1,2), max_df=0.75)
X2 = bigram_vectorizer.fit_transform(df_train.comment)

# first create the vectorizer
# bigram_vectorizer_test = CountVectorizer(ngram_range=(1,2), max_df=0.75)
X2_test = bigram_vectorizer.transform(df_test.comment)

size = X2.shape[0]
size_test = X2_test.shape[0]
# ones_column = np.ones(size).reshape(size, 1)
# new_column = scipy.sparse.csr.csr_matrix(ones_column )
# X_train = scipy.sparse.hstack([new_column,X2])

print('getting caps features ready ...')
# get the new feature
caps_list = []
for i in range(len(df_train)):
    line = df_train.comment[i]
    num = len(filter(lambda x: x in string.uppercase, line))
    den = float(0.1+len(re.findall('[a-zA-Z]', line)))
    ratio = num/den
    caps_list.append(ratio)

caps_list_test = []
for i in range(len(df_test)):
    line = df_test.comment[i]
    num = len(filter(lambda x: x in string.uppercase, line))
    den = float(0.1+len(re.findall('[a-zA-Z]', line)))
    ratio = num/den
    caps_list_test.append(ratio)

print('caps features done...')
# create a np array from it
caps_np = array(caps_list).reshape(size, 1)
new_column = scipy.sparse.csr.csr_matrix(caps_np)
x_train = scipy.sparse.hstack([new_column, X2])

# create a np array from it
caps_np_test = array(caps_list_test).reshape(size_test,1)
new_column_test = scipy.sparse.csr.csr_matrix(caps_np_test)
x_test = scipy.sparse.hstack([new_column_test, X2_test])

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
predicted = model.predict(x_test)
index = 0
f_out = open('../output/feature_1.csv', 'w')

# f_out_only0s = open('../output/output_svm_new_5_0s.csv', 'w')

f_out.write("Id,Category\n")

f_bad = open('../data/bad_words.txt', 'r')
bad_ugrams = f_bad.read().splitlines()

# f_bigram = open('../data/bigram_bad.txt', 'r')
# bad_bigrams = [make_tuple(item) for item in f_bigram.read().splitlines()]

print('Writing to file... \n')
import re
pattern = re.compile("[^\w']")
for item in predicted:
    temp_item = item
    if any(word.lower() in str(df_test.comment[index]).lower() for word in bad_ugrams):
        temp_item = 1
    # map_bgram = Counter(ngrams(pattern.sub(' ',df_test.comment[index].lower()).split(), 2))
    #
    # if any(bigram in map_bgram for bigram in bad_bigrams):
    #     temp_item = 1

    # if any(word.lower() in str(df_test.comment[index]).lower() for word in bad_ugrams):
    #     temp_item = 1

    # if temp_item == 0:
    #     f_out_only0s.write('{0} Label: {1}\n{2}\n\n\n'.format(index, temp_item, df_test.comment[index]))
    f_out.write('{0},{1}\n'.format(index, temp_item))
    index += 1

# df_train['caps'] = df_train.apply(lambda row: len(filter(lambda x: x in string.uppercase, row['comment']))/float(0.1+len(re.findall('[a-zA-Z]',row['comment']))), axis=1)
# df_test['caps'] = df_test.apply(lambda row: len(filter(lambda x: x in string.uppercase, row['comment']))/float(0.1+len(re.findall('[a-zA-Z]',row['comment']))), axis=1)
