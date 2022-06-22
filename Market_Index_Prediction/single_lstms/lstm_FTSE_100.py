import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
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
# print(X_train.shape)
# print(y_train.shape)

model = Sequential()
model.add(LSTM(units = 64,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3)) #The dropout rate is set to 30%, meaning one in 3.33 inputs will be randomly excluded from each update cycle.
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam', metrics=['mean_squared_error'])
model.summary()

# The parameters "batch_size" and "epochs" should be tuned for the particular use case.
# See the GRID Search CV option below.

lstm_pred = model.fit(X_train,y_train,batch_size=30,epochs=40)

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

predicted_stock_price = model.predict(X_test) # Need to understand x.predict() this for the paper
predicted_stock_price = np.reshape(predicted_stock_price, (testing_set_size, 1))
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

print('Number of time-slots:')
print(testing_set_size)

print('Predicted FTSE100 Price:')
# print(predicted_stock_price)
print(predicted_stock_price.transpose())

plt.rcParams["font.family"] = "Times New Roman"
plt.plot(real_stock_price, color='black', label='Real')
plt.plot(predicted_stock_price, color='green', label='Predicted FTSE100')
plt.title('Real vs Predicted FTSE100')
plt.xlabel('Months')
plt.ylabel('FTSE100')
plt.legend()
plt.savefig('lstm_FTSE100.png', dpi=1080, format='png')
plt.show()

# LSTM Evaluation
# Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(real_stock_price, predicted_stock_price)
print(f'Single FTSE100 LSTM MAPE: {MAPE}')



# OPTIONAL: GRID Search (!takes a while!)
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
#
# def build_classifier(optimizer):
#     grid_model = Sequential()
#     grid_model.add(LSTM(units = 64,input_shape=(X_train.shape[1], 1)))
#     grid_model.add(Dropout(0.4))
#     grid_model.add(Dense(1))
#
#     grid_model.compile(loss = 'mse',optimizer = optimizer, metrics = ['mean_squared_error'])
#     return model
#
# grid_model = KerasClassifier(build_fn=build_classifier)
# parameters = {'batch_size' : [10,20,30],
#               'epochs' : [40,50,60,90],
#               'optimizer' : ['adam','Adadelta']}
#
# grid_search  = GridSearchCV(estimator = grid_model,
#                             param_grid = parameters,
#                             n_jobs =-1,
#                             cv = 3)
#
# grid_search = grid_search.fit(X_train,y_train)
#
# print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))