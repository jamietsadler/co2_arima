#!/usr/bin/env python
# coding: utf-8

# # ARIMA Modelling CO2 ppm 
# 
# ## In this notebook I will do some time series analysis on co2 ppm levels over a number of months. I will then fit an arima and sarima model and compare the results.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn


# In[2]:


df = pd.read_csv('datasets/CO2.csv', index_col=0)


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.plot()


# Non Stationary dataset, need to convert into stationary to use arima modelling

# In[6]:


changes = df.pct_change()


# In[7]:


changes.head()


# In[8]:


from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[9]:


plot_acf(df['CO2_ppm'], lags=100)


# In[10]:


from statsmodels.tsa.stattools import adfuller


# In[11]:


result = adfuller(df['CO2_ppm'])


# In[12]:


# Print test statistic
print(result[0])

# Print p-value
print(result[1])

# Print critical values
print(result[4]) 


# Test Statistic & p-value are very high, not stationary

# In[13]:


df_diff = df.diff().dropna()


# In[14]:


df_diff.head()


# In[15]:


result2 = adfuller(df_diff['CO2_ppm'])


# In[16]:


# Print test statistic
print(result2[0])

# Print p-value
print(result2[1])

# Print critical values
print(result2[4])


# p-value now extremely low and time series is stationary

# In[17]:


df_diff.plot()


# In[18]:


plot_acf(df_diff['CO2_ppm'])


# 

# In[19]:


plot_pacf(df_diff['CO2_ppm'])


# In[20]:


co2_train = df.iloc[:int(len(df)*0.95)]
co2_test = df.iloc[int(len(df)*0.95):]


# In[21]:


import pmdarima as pmd
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[22]:


results_arima = pmd.auto_arima(co2_train)


# In[23]:


results_arima.summary()


# In[24]:


model_arima = SARIMAX(co2_train, order=(2,1,1))


# In[25]:


arima_results = model_arima.fit()


# In[26]:


arima_pred = arima_results.get_forecast(steps=len(co2_test))
arima_mean = arima_pred.predicted_mean


# SARIMAX modelling to compare

# In[27]:


results_sarima = pmd.auto_arima(df,
                        seasonal=True,
                        m=12)


# In[28]:


results_sarima.summary()


# In[29]:


results_sarima.plot_diagnostics()


# Good stuff

# In[36]:


model_sarima = SARIMAX(co2_test, order=(2,1,1), seasonal_order=(1,0,1,12))


# In[37]:


sarima_results = model_sarima.fit()


# In[38]:


sarima_pred = sarima_results.get_forecast(steps=len(co2_test))
sarima_mean = sarima_pred.predicted_mean


# In[39]:


dates = co2_test.index


# In[40]:


plt.plot(dates, sarima_mean, label='SARIMA')
plt.plot(dates, arima_mean, label='ARIMA')
plt.plot(co2_test, label='observed')
plt.legend()
plt.show()


# So there we have it, Arima gives ok results with low bias at first, but then soon becomes inaccurate due to not incoporating seasonality into forecasting. 
# Sarima model seems to have a large but consistent bias, but clearly accounts for seasonality in the data

# In[ ]:




