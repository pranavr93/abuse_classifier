from sklearn.base import TransformerMixin


class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


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
from xgboost import *
from sklearn.model_selection import GridSearchCV
import sys
import math
import numpy as np
sys.path.append('xgboost/wrapper/')
import xgboost as xgb
import pickle

class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self


def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in
                    zip(Y_pred, y_true)) / len(Y_pred)


# read the data into pandas data frame
df_train = pd.read_csv('../data/train.tsv', sep='\t', header=0)
df_test = pd.read_csv('../data/test.tsv', sep='\t', header=0)

# parameters = {
#     'clf__eta': [0.05, 0.1, 0.3],
#     'clf__max_depth': [6, 9, 12],  # 12
#     'clf__subsample': [0.9, 1.0],  # 0.9
#     'clf__colsample_bytree': [0.9, 1.0],  # 0.9
# }
# parameters = {
#     'clf__eta': [0.3, 0.4, 0.5],   # 0.5
#     'clf__max_depth': [12, 15, 20],  # 20
#     'clf__subsample': [0.7, 0.8, 0.9],  # 0.7
#     'clf__colsample_bytree': [0.7, 0.9],  # 0.7
# }
# parameters = {
#     'clf__eta': [0.5, 0.6, 0.7],   # 0.5
#     'clf__max_depth': [20, 30, 40],  # 20
#     'clf__subsample': [0.5, 0.6, 0.7],  # 0.7
#     'clf__colsample_bytree': [0.6, 0.7, 0.9],  # 0.7
# }
# parameters = {
#     'clf__eta': [0.5, 0.6, 0.7],   # 0.6
#     'clf__max_depth': [20, 30, 40],  # 40
#     'clf__subsample': [0.5, 0.6, 0.7],  # 0.6
#     'clf__colsample_bytree': [0.6, 0.7, 0.9],  # 0.6
# }
parameters = {
    'clf__eta': [0.3],   # 0.6 good value
    'clf__max_depth': [70],  # 40 increase next
    'clf__subsample': [0.6],  # 0.6 good value
    'clf__colsample_bytree': [0.6],  # 0.6 decrease next
    'clf__eval_metric': ['auc'],
}

pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 7), max_df=0.75)),
                     # ('to_dense', DenseTransformer()),
                     ('clf', XGBoostClassifier(eval_metric='auc',
                                           num_class=2,
                                           nthread=8,
                                           silent=1,
                                           eta=0.3,
                                           max_depth=70,
                                           subsample=0.6,
                                           colsample_bytree=0.5
                                               ))
                     ])
# grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
text_clf = pipeline.fit(df_train.comment, df_train.label)

# # find the parameters that gave the best performance
# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))

filename = '../models/seven_gram_xgboost.sav'
pickle.dump(text_clf, open(filename, 'wb'))

######## Training complete ########

predicted = text_clf.predict(df_train.comment)
# metrics on training data
print('accuracy : {0}'.format(accuracy_score(df_train.label, predicted)))
print('precision : {0}'.format(precision_score(df_train.label, predicted)))
print('recall : {0}'.format(recall_score(df_train.label, predicted)))
print('f1 score : {0}'.format(f1_score(df_train.label, predicted)))

predicted = text_clf.predict(df_test.comment)

# writing results to a file
index = 0
f_out = open('../output/output_xgboosting_10.csv', 'w')
f_out.write("Id,Category\n")

f_bad = open('../data/bad_words.txt', 'r')
bad_list = f_bad.read().splitlines()

for item in predicted:
    temp_item = item
    if any(word in str.lower(df_test.comment[index]) for word in bad_list):
        temp_item = 1

    f_out.write('{0},{1}\n'.format(index, temp_item))
    index += 1
