import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
# Making the Confusion Matrix

def mean_absolute_percentage_error(X_test, y_test): 
    y_pred = regressor.predict(X_test)
    y_pred = sc1.inverse_transform(y_pred)
    y_true = sc1.inverse_transform(y_test)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mad = mean_absolute_error(y_true, y_pred)
    rms = sqrt(mean_squared_error(y_pred, y_true))
    return mad, rms, mape

# Importing the dataset
dataset = pd.read_csv('Train_X.csv', header = None)
label = pd.read_csv('Train_Y.csv', header = None)
X = dataset.iloc[:,:].values
y = label.iloc[:,:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc1 = StandardScaler()
X = sc.fit_transform(X)
y = sc1.fit_transform(y)

# Data processing
X_ = np.zeros((y.shape[0], 100*X.shape[1]))
for repitation in range(X_.shape[0]):
    temp = X[repitation*100:(repitation*100)+100,:]
    temp = np.reshape(temp,(1,-1))
    X_[repitation,:] = temp
X = X_
# check nan
# t = np.isnan(X)


#k = np.sum(t, axis = 1)
# for NaN value 
#X = np.delete(X, [i for i in range(420,428)], 0)
#y = np.delete(y, [i for i in range(420,428)], 0)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

mad_svm, rms_svm, mape_svm = mean_absolute_percentage_error(X_test, y_test)


###########################################
# Fitting Kernel SVM to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(max_depth=2, random_state=0)
regressor.fit(X_train, y_train)

mad_rf, rms_rf, mape_rf = mean_absolute_percentage_error(X_test, y_test)

################################################
# Fitting Kernel SVM to the Training set
from sklearn import neighbors
regressor = neighbors.KNeighborsRegressor(5, weights='distance')
regressor.fit(X_train, y_train)

mad_kn, rms_kn, mape_kn = mean_absolute_percentage_error(X_test, y_test)
 
##################################

#import tensorflow.kerasA
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 12, activation = 'relu', input_shape = (X_train.shape[1],)))


# Adding the second hidden layer
regressor.add(Dense(units = 24, activation = 'relu'))


# Adding the output layer
regressor.add(Dense(units = 1, activation = 'linear'))

# Compiling the ANN
regressor.compile(optimizer = tf.optimizers.Adam(lr=0.0001), loss = 'mse')

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, epochs = 500)


mad_nn, rms_nn, mape_nn = mean_absolute_percentage_error(X_test, y_test)

## saving

Array = np.array([[mad_kn, mad_rf, mad_svm, mad_nn], [rms_kn, rms_rf, rms_svm, rms_nn], [mape_kn, mape_rf, mape_svm, mape_nn]]) 

np.savetxt('result.csv', Array, delimiter=',', fmt='%-7.5f')
# Displaying the array 
#print('Array:\n', Array) 
#file = open("file1.txt", "w+") 
  
# Saving the array in a text file 
#content = str(Array) 
#file.write(content) 
#file.close()