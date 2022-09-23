from pandas import read_csv, concat
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot as plt

df = pd.read_csv('Training_Data_2019_88.csv')
df.fillna(0, inplace=True)

training_data = df.drop('Class',axis=1)
training_output = df['Class']
training_data_cols = np.array(df.drop('Class',axis=1).columns.values.tolist())

# Fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(training_data, training_output)

# Display the relative importance of each attribute
print(model.feature_importances_)
weights = np.array(model.feature_importances_)
training_data_cols_matrix = np.expand_dims(training_data_cols, axis=1)
weights = np.expand_dims(weights, axis=1)

# Table output
table = np.concatenate([training_data_cols_matrix, weights], axis=1)
table = pd.DataFrame(table)
table.columns = ['Attribute', 'Weights']
table.to_csv('weights_importance_uc1.csv', index=False)
print(table)

# Plot output
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})
sorted_idx = model.feature_importances_.argsort()
plt.barh(training_data_cols[sorted_idx], model.feature_importances_[sorted_idx], color='green')
plt.xlabel("Random Forest Feature Importance")
plt.savefig('RFFI_UC1.png', dpi=1080, format='png')
plt.show()
