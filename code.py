#!/usr/bin/env python
# coding: utf-8

#  TP LSTM

# In[2]:


get_ipython().system('pip install pyspark')


# In[3]:


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


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)

from matplotlib import rcParams
sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18,4)})
rcParams['figure.figsize'] = 18,4

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[6]:


rnd_seed=23
np.random.seed=rnd_seed
np.random.set_state=rnd_seed


# In[7]:


spark = SparkSession.builder.master("local[2]").appName("tp").getOrCreate()


# In[8]:


spark


# In[9]:


sc = spark.sparkContext
sc


# In[10]:


sqlContext = SQLContext(spark.sparkContext)
sqlContext


# In[11]:


HOUSING_DATA = '../IOT-temp.csv'


# In[13]:


data = spark.read.csv("../IOT-temp.csv", header=True, inferSchema=True)
data.printSchema()


# In[14]:


data.show()


# In[26]:


df = pd.read_csv(
  "../IOT-temp.csv",
  parse_dates=['noted_date'],
  index_col="noted_date"
)


# In[27]:


df.shape


# In[28]:


df.head()


# In[29]:


df['hour'] = df.index.hour
df['day_of_month'] = df.index.day
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month


# In[30]:


train_size = int(len(df) * 0.9) 
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))


# In[31]:


f_columns = ['id', 'room_id/id', 'temp', 'out/in']


# In[34]:


def create_dataset(X, y, noted_date=1):
    Xs, ys = [], []
    for i in range(len(X) - noted_date):
        v = X.iloc[i:(i + noted_date)].values
        Xs.append(v)
        ys.append(y.iloc[i + noted_date])
    return np.array(Xs), np.array(ys)


# In[35]:


noted_date = 10


# In[36]:


X_train, y_train = create_dataset(train, train.temp, noted_date)
X_test, y_test = create_dataset(test, test.temp, noted_date)


# In[38]:


print(X_train.shape, y_train.shape)


# In[41]:


pip install keras


# In[45]:


import keras


# In[44]:


pip install tensorflow


# 


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


# 


history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)


# 


f_columns = ['id', 'room_id/id', 'temp', 'out/in']


# 


f_transformer = RobustScaler()
f_transformer = f_transformer.fit(train[f_columns].to_numpy())
train.loc[:, f_columns] = f_transformer.transform(
  train[f_columns].to_numpy()
)
test.loc[:, f_columns] = f_transformer.transform(
  test[f_columns].to_numpy()
)


#


temp_transformer = RobustScaler()
temp_transformer = tmp_transformer.fit(train[['temp']])
train['temp'] = temp_transformer.transform(train[['temp']])
test['temp'] = temp_transformer.transform(test[['temp']])

