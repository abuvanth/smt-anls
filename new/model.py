import pandas as pd
import os
import numpy as np
import random
from io import open
from textblob import TextBlob
# encoding=utf8
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
reload(sys)
sys.setdefaultencoding('utf8')
def pos_tag(txt):
    txt=str(txt).encode('ascii', 'ignore').decode('ascii')
    blob=TextBlob(txt)
    tagged=[(word,tag) for word, tag in blob.tags 
                if tag.startswith('NN') or tag.startswith('JJ')]
    return tagged
def wordlist(txt):
    txt=str(txt).encode('ascii', 'ignore').decode('ascii')
    blob=TextBlob(txt)
    wlist=[word for word, tag in blob.tags 
                if tag.startswith('NN') or tag.startswith('JJ')]
    return wlist
print "please wait"
data=pd.read_csv('Amazon_Unlocked_Mobile.csv',encoding='utf-8')
data['processed_Reviews']=data['Reviews'][:10000].apply(pos_tag)

#print processed['processed_Reviews']
bow_transformer = CountVectorizer(analyzer=wordlist).fit(data['Reviews'][:10000])
print len(bow_transformer.vocabulary_)
review=data['Reviews'][:10000]
rating=data['Rating'][:10000]
bow_25 = bow_transformer.transform([data['Reviews'][:10000][24]])
print (bow_25)
print(bow_transformer.get_feature_names()[9042])#Feature extraction
print(bow_transformer.get_feature_names()[10096])
print(bow_transformer.get_feature_names()[2611])
print(bow_transformer.get_feature_names()[10632])
X = bow_transformer.transform(review)
print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)
# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
print('Density: {}'.format((density)))
X_train, X_test, y_train, y_test = train_test_split(X, rating, test_size=0.3, random_state=101)
nb = MultinomialNB()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))
#print(len(data['Reviews'][:10000]))
#for i in range(len(data['Reviews'][:10000])-1):
def get_rating(rvw):
    review_transformed = bow_transformer.transform([rvw])
    return nb.predict(review_transformed)[0]
def get_sentiment(rvw):
    review_transformed = bow_transformer.transform([rvw])
    if nb.predict(review_transformed)[0] >=3:
       return "postive"
    else:
       return "negative"
print ("please Wait.....")
data['new_rating']=data['Reviews'][:10000].apply(get_rating)
data['sentiment']=data['Reviews'][:10000].apply(get_sentiment)
processed_data=data.dropna()#remove null values
processed=processed_data.drop(['Reviews'],axis=1)
processed.to_csv('Sentiments_of_reviews.csv')

