
# coding: utf-8

# In[2]:


# Building a multi class classifier 

# importing the various libraries

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import _pickle as cPickle ### For Python 3
import re


# In[3]:


df= pd.read_csv('sentiment_data.csv')


# In[4]:


#print("FIRST ONE LINE",df.head(1))


# In[5]:


X1=df.iloc[:,9]
print(X1)


# In[6]:


#Y=df['Sentiment']
#print(Y)


# In[7]:


k=[]
for i in df['Sentiment']:
    if i=='Positive':                          #Multi class classifer
        k.append(0)
    elif i=='Neutral':
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


# In[8]:


stop_words = stopwords.words('english') + list(punctuation)
print(stop_words)


# In[9]:


#Let's tokenize the text according to our wish only
tokenizer =RegexpTokenizer(r'\w+')
def tokenizefunc(text):
    words = tokenizer.tokenize(text.lower())
    #lets create a wordlist of tokens 
    wordlist=[word for word in words if word not in stop_words and len(word)>2]
    wrd=[word for word in wordlist if re.match(r"\D",word,re.I)]
   # print(wrd)
    return wrd

#with open(tpath+'tokenize_mcl.pkl','wb') as t:
 #   cPickle.dump(tokenizefunc,t)


# In[10]:


#this has nothing to do wid the code it's just to check the 
o=tokenizefunc("Hi can you plz tell me the status of the following invoices 6789 567tyui 10th?  ']\]")
print(o)         


# In[11]:


print(X1.shape)
print(Y1.shape)
x_train,x_test,y_train,y_test=train_test_split(X1,Y1,test_size=0.2,random_state=0)
#y_train=y_train.astype('int')
#y_test=y_test.astype('int')

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(type(X1))
print(type(Y1))
print(type(x_test))


# In[12]:


cv=TfidfVectorizer(stop_words=stop_words,
                          tokenizer=tokenizefunc)
#cv=TfidfVectorizer(min_df=1,stop_words='english')    


# In[13]:


training_x=cv.fit_transform(x_train)
#print (training_x.toarray)
test_x=cv.transform(x_test)
#print(test_x)


# In[14]:


x=cv.get_feature_names()  # features are the words in our corpus
print(x)


# In[15]:


mnb=MultinomialNB()
mcl=mnb.fit(training_x,y_train)
print(mcl)


# In[16]:


pred=mnb.predict(test_x)
print(pred[1000:2000])


# In[17]:


print(accuracy_score(y_test, pred)*100,"%")


# In[18]:



path ='C:\Python27\SJP\MCL'
tpath = path
with open(tpath+'SJP.pkl', 'wb') as fid:
    cPickle.dump(mcl, fid) 
    fid.close()


# In[19]:


with open(tpath+'SJP.pkl','rb') as fid123:
    clf=cPickle.load(fid123,encoding='bytes') 
#I  don't know how to use pickle
#classification = clf.predict(["hey mann ssly I didn't meant to hurt you"])


# In[20]:


#print(cv.transform("Hi I am all good "))
### NOT USING PICKILING !

df1=pd.Series(["So sorry to you all struggling just live! Praying for relief and strength for you all"])
print(type(df1))


# In[21]:


#print(cv.transform(df1))
c=cv.transform(df1)
print(c.toarray())
cv.inverse_transform(c)


# In[22]:


prednew=mnb.predict(cv.transform(df1))


# In[23]:


print(prednew)


# In[24]:


para_mnb = [{'alpha':[1e-2,1e-3,1e-4],
            'fit_prior' : ['True','False']}]
mnb_clf = GridSearchCV(MultinomialNB(), para_mnb, cv=10)
mnb_clf.fit(training_x,y_train)


# In[25]:


mnb_clf.best_params_


# In[26]:


mnb1=MultinomialNB(alpha= 0.0001, fit_prior= True)
mcl1=mnb1.fit(training_x,y_train)
pred1=mnb.predict(test_x)
print(accuracy_score(y_test, pred1)*100,"%")


# In[27]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print('precision= ',precision_score(y_test, pred1, average=None))
print('recall= ',recall_score(y_test, pred1, average=None))

