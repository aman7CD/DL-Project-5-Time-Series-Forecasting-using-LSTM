
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""DATA  PREPROCESSING"""

training = pd.read_csv('/content/google_stock_prices_train_dataset.csv')

test = pd.read_csv('/content/google_stock_prices_test_dataset.csv')

training_set = training.iloc[:,1:2]
test_set = test.iloc[:,1:2]


training_set = training_set.to_numpy()
test_set = test_set.to_numpy()


from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
scaled_train = mms.fit_transform(training_set)
scaled_test = mms.transform(test_set)

train_x = scaled_train[0:1257]
train_y = scaled_train[1:1258]

test_x = scaled_test[0:122]
test_y = scaled_test[1:123]


#lstm takes inputs: A 3D tensor with shape [batch, timesteps, feature].
train_x = train_x.reshape(1257,1,1)
test_x = test_x.reshape(122,1,1)

"""MODEL BUILDING"""

from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout

from keras.optimizers import Adam
model = Sequential()
model.add(LSTM(units=128, input_shape=(None,1)))

model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='Adam', loss='mean_squared_error', )
model.fit(train_x, train_y, epochs=100, )



"""MAKING THE PREDICTION AND VISUALIZING THE RESULT"""

pred_value = model.predict(test_x)

predicted_stock_prices = mms.inverse_transform(pred_value)

real_stock_prices = mms.inverse_transform(test_y)


"""PLOTTING THE STOCK PRICES"""

plt.plot(real_stock_prices, color='red' ,label='real_stock_prices')
plt.plot(predicted_stock_prices, color='blue',label='predicted_stock_prices')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.legend()

n1 = float(input("Enter today's stock price"))
n1 = np.array(n1)
n1 = n1.reshape(1,-1)
n1 = mms.transform(n1)

n1 = n1.reshape(1,1,1)
n2 = model.predict(n1)

n2 = mms.inverse_transform(n2)
print(f"Tomorrow's stock price will be{n2}" )

