# Recurrent Neural Network
#!pip install numpy
#pip install pandas
#pip install keras



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #to get the dataset,on right side clickon varvariable explorer amd select the files which we want to see(click only once)

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') #for running certain lines of code,select that code and press ctrl+enter to execute. 
training_set = dataset_train.iloc[:, 1:2].values #1:2 as we need numpy array(2 dim),therefore it actually include 1 column

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # we will apply normalization which will give value b/w 0 and 1
training_set_scaled = sc.fit_transform(training_set) #We will not apply Feature Scaling on Test set(ask once ???)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0]) #taking 60 previous data (i.e "open") and calculating data for 61st financial day (y_train) 
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train) #to make it a numpy array 

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #since we are adding one moe dimension on numpy,it is done by reshape().2,3,4 arg. takes batch size, no. of col.(i.e no. of times step), and how many dimension we are adding  respectively
#see once 1st and 2nd argument of bracket that index (see once???)


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout # to add some regularization

# Initialising the RNN
regressor = Sequential() # regressor is used coz we are predicting continous value,i.e regression

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) # 50 correspond to 50 neurons we are taking for first LSTM, Input shape  means the tiny seps and no. of predictor(indicator,i.e 1)
regressor.add(Dropout(0.2)) # to do regularization, we will remove some neurons, 10% of 50, i.e 10

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True)) #we have removed 3rd parameter as now it will be known for him (i.e the no. of inputs) by the neurons
regressor.add(Dropout(0.2)) 

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50)) # since this is the last layer,therefore return_sequence=false and it is removed as it is it's default value also
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # Instead of Adam, can we put here Gradient Descent as C is directly proportional to Y^2, for regression (ask once ???)

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) # 1 epoch mein 1198 data ko train kar rha ( ask once ???) 



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) #axis 0 correspond to vertical concatenation 
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values # here we are putting range i.e from starting day of jan (2017) to last day of jan (2017) .See once,how it got the last date,i.e upper bound (ask once ???)
inputs = inputs.reshape(-1,1) #to convert it into numpy (these parameters are default)
inputs = sc.transform(inputs) # since Feature Scaling is done on the training set,so we need to FS inputs here also but not on test set,coz that are actual values
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0]) #here we are not using Y_test as we want our RNN to predict the dependent variable for test set
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #Inversing the FS value so that it can be converted into actual value  

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()