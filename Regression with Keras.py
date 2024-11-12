import pandas as pd
import numpy as np

import warnings
warnings.simplefilter('ignore', FutureWarning)

# Load dataset
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')

# Check dataset details
concrete_data.head()
print(concrete_data.shape)
print(concrete_data.describe())
print(concrete_data.isnull().sum())

# Prepare predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

# Normalize predictors
predictors_norm = (predictors - predictors.mean()) / predictors.std()
n_cols = predictors_norm.shape[1]

# Import TensorFlow and define the model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define regression model
def regression_model():
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and fit the model
model = regression_model()
history = model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
