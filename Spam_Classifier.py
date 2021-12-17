# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 17:32:18 2021

@author: Tuba
"""

import pandas as pd

path = 'smsspamcollection/SMSSpamCollection'

# read sms csv file with tab delimiter & 2 columns names
messages = pd.read_csv(path, 
                       sep='\t',
                       names=['Label', 'Message'])


#Data cleaning & pre-processing


import re
import nltk
from nltk.corpus import stopwords
# if you apply stemming
from nltk.stem.porter import PorterStemmer
# if you apply lemmatization
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')

# ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# my corpus list
corpus = []

# my stopwords
stop_word = stopwords.words('english')

for i in range(0, len(messages)):
    # remove unnecessary caracters
    review = re.sub('^a-zA-Z', ' ', messages['Message'][i])
    
    # lower my text
    review = review.lower()
    
    # split to get list of words to apply lemmatization
    review = review.split()
    
    # apply lemmatization if not a stop words
    review = [lemmatizer.lemmatize(word) for word in review if not word in stop_word]
    
    # join the base form of words
    review = ' '.join(review)
    
    # append all in my corpus list
    corpus.append(review)
    
    
# Convert my data to a Bag Of words (document matrix)
   
from sklearn.feature_extraction.text import CountVectorizer
# instead of 8428 i'm taking 5000 maximum features  
cv= CountVectorizer(max_features= 5000)

# my input data
X = cv.fit_transform(corpus).toarray()



# My output data (my labels: ham-spam) 

# ham = 1, spam = 0 ==> it's a ham
# ham = 0, spam = 1 ==> it's a spam
y = pd.get_dummies(messages['Label'])

# remove one column, we need just one columns
# if 0 it's spam else it's a spam
y = y.iloc[: , 1].values

# train, test split of x & y
# split my data into training set (80%), & testing (20%)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size= 0.20,
                                                    random_state= 0)

# Train My model using Naive Bayes Classifier
# Naive Bayes works very well with NLP
from sklearn.naive_bayes import MultinomialNB
MultiNB = MultinomialNB()
# Train our data
spam_detector = MultiNB.fit(X_train, y_train)


# make prediction

y_pred = spam_detector.predict(X_test)


# Look at the confusion Matrix
from sklearn.metrics import confusion_matrix

# how many labels are correctly predicted 
cofusion = confusion_matrix(y_test, y_pred)

# Calculate the accuracy of my model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


accuracy = accuracy_score(y_test, y_pred)*100
f1 = f1_score(y_test, y_pred)*100
precision = precision_score(y_test, y_pred)*100
recall = recall_score(y_test, y_pred)*100


print(f'Accuracy_score = {accuracy:.2f}')
print(f'F1_Score = {f1:.2f}')
print(f'Precision_score = {precision:.2f}')
print(f'Recall_score = {recall:.2f}')







