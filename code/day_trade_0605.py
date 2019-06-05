#!/usr/bin/env python
# coding: utf-8

# # $ \Delta $ $上市公司日報酬率標準差_i$ = $a_0$ + $a_1$ * (上市公司現股當沖比重平均)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression
from scipy import stats


# ### 選取上市公司2012/01/06 - 2016/01/06的日報酬率與現股當沖比重資料

# In[2]:


day_trade_data = pd.read_csv("allinformation_20120106_20160106_2.csv")

# find the sepecific time column
#tmp_arr = np.array(list(day_trade_data))
#result = np.where(tmp_arr=='2014/1/6')
#print(result[0])

# 把dataframe切割成兩部分(before 2014/01/06 和 after 2014/01/06)
day_trade_data_before = day_trade_data.iloc[:,:496] # 2012/01/06 - 2014/01/06
day_trade_data_after = day_trade_data.iloc[:,496:] # 2014/01/07 - 2016/01/06
#day_trade_data_number_and_compnames = day_trade_data.iloc[:,:2]


# In[3]:


day_trade_data_before.head(n=5)


# In[4]:


day_trade_data_after.head(n=5) 


# In[33]:


# 計算每間公司的日報酬率標準差
day_trade_data_before['company_std_before'] = day_trade_data_before.std(axis=1)
day_trade_data_after['company_std_after'] = day_trade_data_after.std(axis=1)

# 計算每間公司的現股當沖比重平均
day_trade_data_before['company_day_mean_before'] = day_trade_data_before.mean(axis=1)
day_trade_data_after['company_mean_after'] = day_trade_data_after.mean(axis=1)

x = day_trade_data_after.iloc[1::2]['company_mean_after']
x2 = day_trade_data_after.iloc[1::2]['company_mean_after']
y = day_trade_data_after.iloc[0::2]['company_std_after'] - day_trade_data_before.iloc[0::2]['company_std_before']


# In[23]:


type(x)


# In[24]:


type(y)


# ### $ \Delta $ $上市公司日報酬率標準差_i$ = 0.1606 -0.0015 * (上市公司現股當沖比重平均)

# In[36]:


y = list(y)
#x = sm.add_constant(x)
results = sm.OLS(y,x).fit()


# In[37]:


print(results.summary())


# ### https://www.statsmodels.org/stable/index.html
# ### https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression

# In[ ]:




