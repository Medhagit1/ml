
# coding: utf-8

# In[94]:


# Building a multi class classifier 

# importing the various libraries

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[95]:


df= pd.read_csv('sentiment_data.csv')


# In[96]:


#print("FIRST ONE LINE",df.head(1))


# In[111]:


X1=df.iloc[:,9]
print(X1)


# In[98]:


#Y=df['Sentiment']
#print(Y)


# In[113]:


k=[]
for i in df['Sentiment']:
    if i=='Positive':
        k.append(0)
    elif i=='Negitive':
        k.append(2)
    else:
        k.append(1)
df['Output']=k   # ?  # why created one more label
Y1=df['Output']
#print(Y1)
print(X1.shape)
print(Y1.shape)
#print(df['Output']) 
#print(df.describe)


# In[100]:


stop_words = stopwords.words('english') + list(punctuation)
print(stop_words)


# In[101]:


#Let's tokenize the text according to our wish only
tokenizer =RegexpTokenizer(r'\w+')
def tokenizefunc(text):
    words = tokenizer.tokenize(text.lower())
    #lets create a wordlist of tokens 
    wordlist=[word for word in words if word not in stop_words and len(word)>2]
    return wordlist


# In[102]:


#this has nothing to do wid the code it's just to check the 
o=tokenizefunc("Hi can you plz tell me the status of the following invoices?  ']\]")
print(o)         


# In[120]:


print(X1.shape)
print(Y1.shape)
x_train,x_test,y_train,y_test=train_test_split(X1,Y1,test_size=0.2,random_state=0)
y_train=y_train.astype('int')
y_test=y_test.astype('int')

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(type(X1))
print(type(Y1))


# In[121]:


cv=TfidfVectorizer(stop_words=stop_words,
                            tokenizer=tokenizefunc)


# In[122]:


training_x=cv.fit_transform(x_train)
test_x=cv.transform(x_test)


# In[123]:


mnb=MultinomialNB()
mnb.fit(training_x,y_train)


# In[124]:


pred=mnb.predict(test_x)
print(pred.shape)


# In[125]:


print(accuracy_score(y_test, pred)*100,"%")

