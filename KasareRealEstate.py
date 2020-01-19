#!/usr/bin/env python
# coding: utf-8

# # Kasare Real Estate

# In[1]:


import pandas as pd

housing = pd.read_csv("data.csv")


# In[2]:


housing.head()


# In[3]:


housing.info()


# In[4]:


housing['CHAS'].value_counts()


# In[5]:


housing.describe()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


housing.hist(bins=50, figsize=(20, 15))


# ## Train-Test Splitting

# In[9]:


#for learning purpose
import numpy as np
def split_train_test(data, test_ratio):
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[10]:


train_set, test_set = split_train_test(housing, 0.2)


# In[11]:


print(f"Rows is train set:{len(train_set)}\nRows n test set {len(test_set)}")


# In[12]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_test_set['CHAS'].value_counts()


# In[15]:


strat_train_set['CHAS'].value_counts()


# ## Looking for correlation

# In[16]:


corr_matrix = housing.corr()


# In[17]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[18]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12, 8))


# In[19]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# ## Trying attribute combination

# In[20]:


housing["TAXRM"] = housing["TAX"]/housing["RM"]


# In[21]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[22]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing Attributes

# In[23]:


# To take care of missing attributes you have three options:
#     1.get rid of the missing data points
#     2.get rid of whole attribute
#     3.set the value to some value


# In[24]:


a = housing.dropna(subset=["RM"]) #Option1
a.shape


# In[25]:


housing.drop("RM", axis=1).shape #option2
#note that there is no RM column and also note that the original housing dataframe will remain unchanged


# In[26]:


median = housing["RM"].median()


# In[27]:


housing ["RM"].fillna(median) #Option3
#note that the original housing dataframe will remain unchanged


# In[28]:


housing.shape


# In[29]:


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
imputer.fit(housing)


# In[30]:


imputer.statistics_


# In[31]:


x = imputer.transform(housing)


# In[32]:


housing_tr = pd.DataFrame(x, columns=housing.columns)


# In[33]:


housing_tr.describe()


#  ## SciKit Learn

# Primarily, three types of objects
# 1.Estimator - its estimates some parameter based on dataset. Eg Imputer
# It has a fit model and transform method.
# Fit method - Fit the dataset and calculates internal parameters
# 
# 2.Transfomers - transforms method takes input and returns output based on learnings from fit()
# 
# 3.Predictors - LinearRegression model is an example of predictors. fit() and predict() are two common functions. It also gives score() function which will evaluate predictions

# ## Creating a Pipeline

# In[34]:


from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

my_pipeline = Pipeline([
    ('imputer',Imputer(strategy='median')),
    ('std_scalar',StandardScaler()),
])


# In[35]:


housing_tr_num = my_pipeline.fit_transform(housing)


# In[36]:


housing_tr_num


# In[37]:


housing_tr_num.shape


# ## Selecting a desired model for Kasare Real Estates

# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_tr_num, housing_labels)


# In[39]:


some_data = housing.iloc[:5]


# In[40]:


some_labels = housing_labels.iloc[:5]


# In[41]:


prepared_data = my_pipeline.transform(some_data)


# In[42]:


model.predict(prepared_data)


# In[43]:


list(some_labels)


# ## Evaluating the model

# In[44]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_tr_num)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[45]:


rmse


# ## Using cross validation technique

# In[46]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_tr_num, housing_labels, scoring="neg_mean_squared_error")
rmse_scores = np.sqrt(-scores)


# In[47]:


rmse_scores


# In[48]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[49]:


print_scores(rmse_scores)


# In[ ]:




