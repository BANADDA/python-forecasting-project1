import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns

#Load Data_Set
#----------------------------------------------------
#----------------------------------------------------
df = pd.read_csv(r'C:\Users\BANADDA MUBARAKA\Desktop\C PROGRAMS\AirPassengers.csv')

#Converting Month from text to datetime by overwriting the column
df['Month'] = pd.to_datetime(df['Month'])
print(df.dtypes)
print('')
print(df.head())
 
#Set Index to Month
df.set_index('Month', inplace = True)
print(df.head())

#Plot Passengers
plt.plot(df['#Passengers'])
#----------------------------------------------------
#----------------------------------------------------

#Checking for Data Stationality
from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest = adfuller(df)
print('pvalue = ', pvalue, "If above 0.05, data is not stationary")
#----------------------------------------------------
#----------------------------------------------------

# fit model
model = ARIMA(df, order=(5,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()