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

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                      ('clf', LinearSVC())
])
text_clf = text_clf.fit(df_train.comment, df_train.label)

######## Training complete ########

predicted = text_clf.predict(df_train.comment)
for i in range(len(predicted)):
    if predicted[i] != df_train.label[i]:
        print('Correct answer is {0}'.format(df_train.label[i]))
        print(df_train.comment[i])
        print()
        print()
# metrics on training data
print('accuracy : {0}'.format(accuracy_score(df_train.label, predicted)))
print('precision : {0}'.format(precision_score(df_train.label, predicted)))
print('recall : {0}'.format(recall_score(df_train.label, predicted)))
print('f1 score : {0}'.format(f1_score(df_train.label, predicted)))

predicted = text_clf.predict(df_test.comment)

# writing results to a file
index = 0
f_out = open('../output/output_svm2.csv', 'w')
f_out.write("Id,Category\n")
for item in predicted:
    f_out.write(u'{0},{1}\n'.format(index, item))
    index += 1