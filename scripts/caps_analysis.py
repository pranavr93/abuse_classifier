import string
import re
import pandas as pd
import collections
df_train = pd.read_csv('../data/test.tsv', sep='\t', header=0)
good = 0
bad = 0
od = collections.OrderedDict()
f_out = open('../analysis/caps_2.txt','w')

for i in range(len(df_train)):
    line = df_train.comment[i]
    #label = df_train.label[i]
    num = len(filter(lambda x: x in string.uppercase, line))
    den = float(0.1+len(re.findall('[a-zA-Z]', line)))
    ratio = num/den
    # if ratio > 0.95 and label == 1:
    #     good+=1
    # elif ratio > 0.95 and label ==0:
    #     bad+=1
    od[ratio] = df_train.comment[i]

# print(good, bad)
for k, v in sorted(od.iteritems()):
    f_out.write('{0}\n{1}\n\n'.format(k, v))
    # f_out.write('{0} : {1}\n'.format(k, v))