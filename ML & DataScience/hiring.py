import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n
import math

df = pd.read_csv("E:\MARVELLOUS INFOSYSTEM\ML & DataScience\hiring.csv")
print(df)

df.experience = df.experience.fillna("Zero")
print(df)

df.experience = df.experience.apply(w2n.word_to_num)
print(df)

median_test_score = math.floor(df['test_score(out of 10)'].mean())
print(median_test_score)

df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)
print(df)

reg = linear_model.LinearRegression()

reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])
print(reg)

Res1 = reg.predict([[2,9,6]])
Res2 = reg.predict([[12,10,10]])

print(Res1)
print(Res2)

