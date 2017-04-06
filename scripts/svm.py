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
# read the data into pandas data frame
df_train = pd.read_csv('../data/train.tsv', sep='\t', header=0)
df_test = pd.read_csv('../data/test.tsv', sep='\t', header=0)

# pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), max_df=0.75)),
#                       ('clf', LinearSVC(random_state=0))
# ])

# pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.75)),
#                       ('clf', LinearSVC(random_state=0, max_iter=2000, C=0.5, class_weight='balanced'))
# ]) 0.75695

pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.75)),
                      ('clf', LinearSVC(random_state=0, max_iter=4000, C=0.05, class_weight='balanced'))
])



text_clf = pipeline.fit(df_train['comment'], df_train.label)

######## Training complete ########

predicted = text_clf.predict(df_train['comment'])
# metrics on training data
print('accuracy : {0}'.format(accuracy_score(df_train.label, predicted)))
print('precision : {0}'.format(precision_score(df_train.label, predicted)))
print('recall : {0}'.format(recall_score(df_train.label, predicted)))
print('f1 score : {0}'.format(f1_score(df_train.label, predicted)))

predicted = text_clf.predict(df_test['comment'])

# writing results to a file
index = 0
f_out = open('../output/svm_10.csv', 'w')
f_out.write("Id,Category\n")

f_bad = open('../data/bad_words.txt', 'r')
bad_ugrams = f_bad.read().splitlines()

f_bigram = open('../data/bigram_bad.txt', 'r')
bad_bigrams = [make_tuple(item) for item in f_bigram.read().splitlines()]

print('Writing to file... \n')
import re

f_out_only0s = open('../analysis/svm_0s.csv','w')
pattern = re.compile("[^\w']")
for item in predicted:
    temp_item = item

    # map_ugram = Counter(ngrams(pattern.sub(' ',df_test.comment[index].lower()).split(), 1))
    # if any(ugram in map_ugram for ugram in bad_ugrams):
    #     temp_item = 1

    # map_bgram = Counter(ngrams(pattern.sub(' ',df_test.comment[index].lower()).split(), 2))
    #
    # if any(bigram in map_bgram for bigram in bad_bigrams):
    #     temp_item = 1

    if any(word.lower() in str(df_test.comment[index]).lower() for word in bad_ugrams):
        temp_item = 1

    # total_len = len(df_test.comment[index])
    # upper_len = len(filter(lambda x: x in string.uppercase, df_test.comment[index]))
    # ratio = upper_len/float(total_len)
    # if ratio > 0.4:
    #     temp_item = 1
    if temp_item == 0:
        f_out_only0s.write('{0} Label: {1}\n{2}\n\n\n'.format(index, temp_item, df_test.comment[index]))
    f_out.write('{0},{1}\n'.format(index, temp_item))
    index += 1