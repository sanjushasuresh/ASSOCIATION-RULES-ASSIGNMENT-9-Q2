# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:01:46 2022

@author: SANJUSHA
"""

import numpy  as np
import pandas as pd

df=pd.read_csv("my_movies.csv")
df.head()
df.columns
df=df.iloc[:,0:5]
df
df.shape

# To use Apriori Algorithm the data should be in list
Movies=[]
for i in range(0, 10):
  Movies.append([str(df.values[i,j]) for j in range(0, 5)])
Movies
type(Movies)
len(Movies)

from apyori import apriori
rules=apriori(transactions=Movies, min_support = 0.02, min_confidence = 0.3, min_lift = 2, min_length = 3, max_length=4)
rules
results=list(rules)
results

def inspect(results):
    movie1         = [tuple(result[2][0][0])[0] for result in results]
    movie2       = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(movie1,movie2, supports, confidences, lifts))

resultsdf = pd.DataFrame(inspect(results), columns = ['movie1', 'movie2', 'Support', 'Confidence', 'Lift'])
resultsdf

resultsdf.nlargest(n=20, columns='Lift')

# Inference :
    
# for min_support = 0.03, min_confidence = 0.2, min_lift = 2, min_length = 2, max_length=2, 8 Rules are generated
# for min_support = 0.003, min_confidence = 0.3, min_lift = 2, min_length = 2, max_length=2, 8 Rules are generated
# for min_support = 0.03, min_confidence = 0.4, min_lift = 2, min_length = 2, max_length=2, 8 Rules are generated
# Here, Green Mile and LOTR, Harry Potter1 and Harry Potter2, LOTR1 and LOTR2 each has the highest lift of 5
# Even though the values of support and confidence are changed, the number of rules remain the same.

# Changing min length
# for min_support = 0.03, min_confidence = 0.4, min_lift = 2, min_length = 3, max_length=3, 25 Rules are generated
# for min_support = 0.03, min_confidence = 0.4, min_lift = 2, min_length = 4, max_length=4, 35 Rules are generated
# for min_support = 0.02, min_confidence = 0.3, min_lift = 2, min_length = 3, max_length=4, 35 Rules are generated
# Here also the highest are the above combinations with lift 5 and also Green Mile has the most association with other movies

