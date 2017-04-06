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
from sklearn.svm import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# if true, does split, else validates on train set
TRAIN_SPLIT_VALIDATION = True
TRAIN_SPLIT = 0.5   # set to 0.2 under normal circumstances

TRAIN_DATA = '../data/train.tsv'
TEST_DATA = '../data/test.tsv'
OUTPUT_FILE = '../output/output_svm_experiment_gsearch4.csv'

# read the data into pandas data frame
df = pd.read_csv(TRAIN_DATA, sep='\t', header=0)
df_test = pd.read_csv(TEST_DATA, sep='\t', header=0)

pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.7)),
                      ('clf', LinearSVC(random_state=0, max_iter=5000, class_weight='balanced', C=0.1))
])
parameters = {
    # 'clf__class_weight': ('balanced', None),
    # 'clf__C': (0.1, 0.15, 0.2),
    # 'vect__max_df': (0.75, 0.65, 0.85)
    'vect__lowercase': (True, False)
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

if TRAIN_SPLIT_VALIDATION:
    df_train, df_train_test = train_test_split(df, test_size=TRAIN_SPLIT)
else:
    df_train = df_train_test = df

print('Fitting training data with row count : {0}'.format(len(df_train)))
text_clf = grid_search.fit(df_train.comment, df_train.label)

# find the parameters that gave the best performance
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

######## Training complete ########

# metrics on training data (80-20)/ full
predicted = text_clf.predict(df_train_test.comment)
if TRAIN_SPLIT_VALIDATION:
    print('metrics on 20% of the training data set:')
else:
    print('metrics on 100% of the training data set:')

print('accuracy : {0}'.format(accuracy_score(df_train_test.label, predicted)))
print('precision : {0}'.format(precision_score(df_train_test.label, predicted)))
print('recall : {0}'.format(recall_score(df_train_test.label, predicted)))
print('f1 score : {0}'.format(f1_score(df_train_test.label, predicted)))

########### ACTING ON ACTUAL TEST FILE ###############
# writing results to a file
predicted = text_clf.predict(df_test.comment)
index = 0
f_out = open(OUTPUT_FILE, 'w')
f_out.write("Id,Category\n")
for item in predicted:
    f_out.write('{0},{1}\n'.format(index, item))
    index += 1


def wrong_answers(predicted, actual):
    for i in range(len(predicted)):
        if predicted[i] != actual[i]:
            print('Correct answer is {0}'.format(actual[i]))
            print(df_train.comment[i])
            print()
            print()
