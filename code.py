#!/usr/bin/env python
# coding: utf-8

# In[7]:


from google.colab import drive  
drive.mount('/content/drive')


# In[6]:


get_ipython().system('pip install pyspark')


# In[8]:



import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator




import seaborn as sns
import matplotlib.pyplot as plt




from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)

from matplotlib import rcParams
sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18,4)})
rcParams['figure.figsize'] = 18,4

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")




rnd_seed=23
np.random.seed=rnd_seed
np.random.set_state=rnd_seed




spark = SparkSession.builder.master("local[2]").appName("tp").getOrCreate()




spark




sc = spark.sparkContext
sc




sqlContext = SQLContext(spark.sparkContext)
sqlContext




HOUSING_DATA = '/content/drive/MyDrive/IOT-temp.csv'




data = spark.read.csv("/content/drive/MyDrive/IOT-temp.csv", header=True, inferSchema=True)
data.printSchema()
data.show()


# In[10]:






df = pd.read_csv(
  '/content/drive/MyDrive/IOT-temp.csv',
  parse_dates=['noted_date'],
  index_col="noted_date"
)


# In[11]:


df.shape


# In[12]:


df.head()


# In[13]:




df['hour'] = df.index.hour
df['day_of_month'] = df.index.day
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month




train_size = int(len(df) * 0.9) 
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))


# In[14]:



f_columns = ['id', 'room_id/id', 'temp', 'out/in']




def create_dataset(X, y, noted_date=1):
    Xs, ys = [], []
    for i in range(len(X) - noted_date):
        v = X.iloc[i:(i + noted_date)].values
        Xs.append(v)
        ys.append(y.iloc[i + noted_date])
    return np.array(Xs), np.array(ys)




noted_date = 10




X_train, y_train = create_dataset(train, train.temp, noted_date)
X_test, y_test = create_dataset(test, test.temp, noted_date)




print(X_train.shape, y_train.shape)


# In[15]:


import keras
model = keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=128,
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
)
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[21]:


from sklearn.preprocessing import RobustScaler


# In[ ]:


history = model.fit(
    X_train, y_train
   )


# In[ ]:


f_columns = ['id', 'room_id/id', 'temp', 'out/in']
f_transformer = RobustScaler()
f_transformer = f_transformer.fit(train[f_columns].to_numpy())
train.loc[:, f_columns] = f_transformer.transform(
  train[f_columns].to_numpy()
)
test.loc[:, f_columns] = f_transformer.transform(
  test[f_columns].to_numpy()
)

temp_transformer = RobustScaler()
temp_transformer = tmp_transformer.fit(train[['temp']])
train['temp'] = temp_transformer.transform(train[['temp']])
test['temp'] = temp_transformer.transform(test[['temp']])

