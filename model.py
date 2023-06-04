import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns

df_train = pd.read_csv('./train.csv', index_col=0)
df_test = pd.read_csv('./test.csv', index_col=0)

df_train['school'] = df_train['school'].map({'GP': 1, 'MS': 2})
df_train['class'] = df_train['class'].map({'por': 1, 'mat': 2})
df_train['sex'] = df_train['sex'].map({'F': 1, 'M': 2})
df_train['address'] = df_train['address'].map({'U': 1, 'R': 2})
df_train['famsize'] = df_train['famsize'].map({'GT3': 1, 'LE3': 2})
df_train['Pstatus'] = df_train['Pstatus'].map({'T': 1, 'A': 2})
df_train['Mjob'] = df_train['Mjob'].map(
    {'other': 1, 'services': 2, 'at_home': 3, 'teacher': 4, 'health': 5})
df_train['Fjob'] = df_train['Fjob'].map(
    {'other': 1, 'services': 2, 'teacher': 3, 'at_home': 4, 'health': 5})
df_train['reason'] = df_train['reason'].map(
    {'course': 1, 'reputation': 2, 'home': 3, 'other': 4})
df_train['guardian'] = df_train['guardian'].map(
    {'mother': 1, 'father': 2, 'other': 3})

print(df_train)
