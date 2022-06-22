import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Input, Dense
from keras.models import Model
from single_lstms.variables import testing_set_size

dataset_train = pd.read_csv('Processed_Input_Data_FTSE100_1985_21.csv', header=0, index_col=0)
training_set = dataset_train.iloc[:, 0:1].values

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(0, (len(training_set) - testing_set_size)):
    X_train.append(training_set_scaled[i:i+1, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

inputs = Input(shape=(X_train.shape[1], 1))
x = LSTM(50, return_sequences=True)(inputs)
x = Dropout(0.2)(x)
x = LSTM(50, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(50, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = Dense(1, activation="linear")(x)
x = Model(inputs=inputs, outputs=x)
x.compile(optimizer='adam', loss='mean_squared_error')
x.fit(X_train, y_train, epochs=90, batch_size=10) # Need to understand x.fit() this for the paper

dataset_test = pd.read_csv('Processed_Input_Data_FTSE100_1985_21.csv', header=0, index_col=0)
real_stock_price = dataset_test.iloc[(len(training_set) - testing_set_size):, 0:1].values

dataset_total = pd.concat((dataset_train['FTSE100'], dataset_test['FTSE100']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test):].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range((len(training_set) - testing_set_size), len(training_set)):
    X_test.append(inputs[i:i+1, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = x.predict(X_test) # Need to understand x.predict() this for the paper
predicted_stock_price = np.reshape(predicted_stock_price, (testing_set_size, 1))
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

print('Number of time-slots:')
print(testing_set_size)

print('Predicted FTSE100 Price:')
print(predicted_stock_price)

plt.rcParams["font.family"] = "Times New Roman"
plt.plot(real_stock_price, color='black', label='Real')
plt.plot(predicted_stock_price, color='green', label='Predicted FTSE100')
plt.title('Real vs Predicted FTSE100')
plt.xlabel('Months')
plt.ylabel('FTSE100')
plt.legend()
plt.savefig('second_option_lstm_FTSE100.png', dpi=1080, format='png')
plt.show()