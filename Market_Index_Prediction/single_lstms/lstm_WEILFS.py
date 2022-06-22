import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from single_lstms.variables import testing_set_size

dataset_train = pd.read_csv('Processed_Input_Data_FTSE100_1985_21.csv', header=0, index_col=0)
training_set = dataset_train.iloc[:, 9:10].values

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(0, (len(training_set) - testing_set_size)):
    X_train.append(training_set_scaled[i:i+1, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units = 64,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3)) #The dropout rate is set to 30%, meaning one in 3.33 inputs will be randomly excluded from each update cycle.
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam', metrics=['mean_squared_error'])
model.summary()

# The parameters "batch_size" and "epochs" should be tuned for the particular use case.

mei_lstm_pred = model.fit(X_train,y_train,batch_size=10,epochs=70)
dataset_test = pd.read_csv('Processed_Input_Data_FTSE100_1985_21.csv', header=0, index_col=0)
real_stock_price = dataset_test.iloc[(len(training_set) - testing_set_size):, 9:10].values

dataset_total = pd.concat((dataset_train['WEILFS'], dataset_test['WEILFS']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test):].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range((len(training_set) - testing_set_size), len(training_set)):
    X_test.append(inputs[i:i+1, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = np.reshape(predicted_stock_price, (testing_set_size, 1))
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#it's called a predicted_stock_price for our convenience here, but it's the indicator's value

print('Number of time-slots:')
print(testing_set_size)

print('Predicted WEILFS:')
# print(predicted_stock_price)
print(predicted_stock_price.transpose())

plt.rcParams["font.family"] = "Times New Roman"
plt.plot(real_stock_price, color='black', label='Real')
plt.plot(predicted_stock_price, color='green', label='Predicted WEILFS')
plt.title('Real vs Predicted WEILFS')
plt.xlabel('Months')
plt.ylabel('WEILFS')
plt.legend()
plt.savefig('lstm_WEILFS.png', dpi=1080, format='png')
plt.show()