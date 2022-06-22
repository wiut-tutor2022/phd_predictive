import cypher_queries_uc1 as cq
import pandas as pd
import numpy as np

Training_Data = pd.read_csv("Training_Data_2019_43.csv")
Training_Data_columns = pd.read_csv("Training_Data_2019_43.csv").columns.tolist()
ar_node_name_index = Training_Data_columns.index(cq.ar_node_name)
fs_node_name_index = Training_Data_columns.index(cq.fs_node_name)
lr_node_name_index = Training_Data_columns.index(cq.lr_node_name)
pr_node_name_index = Training_Data_columns.index(cq.pr_node_name)

feature_selection_column_indexes = [ar_node_name_index, fs_node_name_index, lr_node_name_index, pr_node_name_index]

print('Feature Selection Column Indexes:')
print(feature_selection_column_indexes)

Training_Data_columns_with_feature_selection = cq.max_weighted_features
Training_Data_columns_with_feature_selection = np.array(Training_Data_columns_with_feature_selection)
Training_Data_columns_with_feature_selection = np.reshape(Training_Data_columns_with_feature_selection, (len(Training_Data_columns_with_feature_selection), 1))
Training_Data_columns_with_feature_selection = Training_Data_columns_with_feature_selection.T
print(Training_Data_columns_with_feature_selection)
Training_Data_with_feature_selection = Training_Data.iloc[:, feature_selection_column_indexes].values
concatenated = np.concatenate([Training_Data_columns_with_feature_selection, Training_Data_with_feature_selection])
print(concatenated)

#Store these in a CSV file
np.savetxt("Training_Data_with_Neo4j_Feature_Selection.csv", concatenated, delimiter=",", comments='', fmt='%s')