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

# read the data into pandas data frame
df_train = pd.read_csv('../data/train.tsv', sep='\t', header=0)
df_test = pd.read_csv('../data/test.tsv', sep='\t', header=0)

pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), max_df=0.75)),
                      ('clf', LinearSVC(random_state=0))
])

pipeline2 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.75)),
                      ('clf', MultinomialNB())
])

text_clf_svm = pipeline.fit(df_train.comment, df_train.label)
text_clf_nb = pipeline.fit(df_train.comment, df_train.label)

######## Training complete ########

predicted_svm = text_clf_svm.predict(df_test.comment)
predicted_nb = text_clf_nb.predict(df_test.comment)

# writing results to a file
index = 0
f_out = open('../output/output_hybrid.csv', 'w')
f_out.write("Id,Category\n")

f_bad = open('../data/bad_words.txt', 'r')
bad_list = f_bad.read().splitlines()

for item in predicted_svm:
    temp_item = item
    if any(word in str.lower(df_test.comment[index]) for word in bad_list):
        temp_item = 1

    if temp_item == 0:
        temp_item += predicted_nb[index]

    f_out.write('{0},{1}\n'.format(index, temp_item))

    index += 1