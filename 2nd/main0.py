import datetime
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from matplotlib import pyplot as plt
import seaborn as sns
import os
import changefinder
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
import shap
shap.initjs()
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from tabulate import tabulate
from IPython.display import HTML, display
timestamp = datetime.datetime.fromtimestamp(1451624400)
print(timestamp)

df = pd.read_csv('../HomeC.csv', low_memory=False)

df['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(df),  freq='min')) #upgrade the time column with a readable date
df['year'] = df['time'].apply(lambda x : x.year)
df['month'] = df['time'].apply(lambda x : x.month)
df['day'] = df['time'].apply(lambda x : x.day)
df['weekday'] = df['time'].apply(lambda x : x.weekday())
df['weekofyear'] = df['time'].apply(lambda x : x.weekofyear)
df['hour'] = df['time'].apply(lambda x : x.hour)
df['minute'] = df['time'].apply(lambda x : x.minute)