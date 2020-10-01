#!/usr/bin/env python
# coding: utf-8

# In[4]:

# Hack


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


customer = pd.read_csv('Ecommerce Customers')


# In[8]:


customer.head()


# In[9]:


customer.info()


# In[11]:


customer.describe()


# In[13]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customer)


# In[14]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent', data=customer)


# In[16]:


sns.jointplot(x='Time on App',y='Length of Membership',data=customer,kind='hex')


# In[17]:


sns.pairplot(customer)


# In[19]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customer)


# In[21]:


customer.columns


# In[25]:


y = customer['Yearly Amount Spent']


# In[26]:


X = customer[[ 'Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]


# In[27]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=101)


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


lm = LinearRegression()


# In[32]:


lm.fit(X_train,y_train)


# In[33]:


lm.coef_


# In[34]:


predictions = lm.predict(X_test)


# In[37]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test (true values)')
plt.ylabel('Predicted Values')


# In[38]:


from sklearn import metrics


# In[39]:


print('MAE',metrics.mean_absolute_error(y_test,predictions))
print('MSE',metrics.mean_squared_error(y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[40]:


metrics.explained_variance_score(y_test,predictions)


# In[41]:


sns.distplot(y_test-predictions,bins=50)


# In[44]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coef'])
cdf


# In[1]:


# now according to me company should focus on its website, or it should enhance it's app so that it can excel more !


# In[ ]:




