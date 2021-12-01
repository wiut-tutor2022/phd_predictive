from neo4j import GraphDatabase
from owlready2 import *
import pandas as pd
import numpy as np

class Neo4jDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_macroeconomic_indicator_nodes(self, ontology):
        with self.driver.session() as session:
            session.run("MATCH (n) WHERE ")


# DB password is last parameter of HelloWorldExample() method (11111)
neo4j_db = Neo4jDB("bolt://localhost:7687", "neo4j", "000000")
onto_path.append(".")
onto = get_ontology("OBP_Ontology_v.2.1.owl")
onto.load()

# Get the weightiest nodes
Activity_Ratios_node = ''
Financial_Sustainability_node = ''
Liquidity_Ratios_node = ''
Profitability_Ratios_node = ''
with neo4j_db.driver.session() as session:
    Activity_Ratios_node = session.run("""MATCH (n1:Class {name: "Activity_Ratios"})<-[r1:subclass_of]-(scn1:Class)
                WITH max(scn1.Weight) AS maximum
                MATCH (n2:Class {name: "Activity_Ratios"})<-[r2:subclass_of]-(scn2:Class)
                WHERE scn2.Weight = maximum
                RETURN scn2""").data()

    Financial_Sustainability_node = session.run("""MATCH (n1:Class {name: "Financial_Sustainability"})<-[r1:subclass_of]-(scn1:Class)
                WITH max(scn1.Weight) AS maximum
                MATCH (n2:Class {name: "Financial_Sustainability"})<-[r2:subclass_of]-(scn2:Class)
                WHERE scn2.Weight = maximum
                RETURN scn2""").data()

    Liquidity_Ratios_node = session.run("""MATCH (n1:Class {name: "Liquidity_Ratios"})<-[r1:subclass_of]-(scn1:Class)
                WITH max(scn1.Weight) AS maximum
                MATCH (n2:Class {name: "Liquidity_Ratios"})<-[r2:subclass_of]-(scn2:Class)
                WHERE scn2.Weight = maximum
                RETURN scn2""").data()

    Profitability_Ratios_node = session.run("""MATCH (n1:Class {name: "Profitability_Ratios"})<-[r1:subclass_of]-(scn1:Class)
                WITH max(scn1.Weight) AS maximum
                MATCH (n2:Class {name: "Profitability_Ratios"})<-[r2:subclass_of]-(scn2:Class)
                WHERE scn2.Weight = maximum
                RETURN scn2""").data()


ar_node_name = Activity_Ratios_node[0]["scn2"]["name"]
fs_node_name = Financial_Sustainability_node[0]["scn2"]["name"]
lr_node_name = Liquidity_Ratios_node[0]["scn2"]["name"]
pr_node_name = Profitability_Ratios_node[0]["scn2"]["name"]

max_weighted_features = [
    ar_node_name,
    fs_node_name,
    lr_node_name,
    pr_node_name,
]

print('Max Weighted Features (1 for each Category):')
print(max_weighted_features)


Ratios_Export = pd.read_csv("Ratios_Export.csv")
Ratios_Export_columns = pd.read_csv("Ratios_Export.csv").columns.tolist()
ar_node_name_index = Ratios_Export_columns.index(ar_node_name)
fs_node_name_index = Ratios_Export_columns.index(fs_node_name)
lr_node_name_index = Ratios_Export_columns.index(lr_node_name)
pr_node_name_index = Ratios_Export_columns.index(pr_node_name)

feature_selection_column_indexes = [ar_node_name_index, fs_node_name_index, lr_node_name_index, pr_node_name_index]

print('Feature Selection Column Indexes:')
print(feature_selection_column_indexes)
print(max_weighted_features)
Ratios_Export_columns_with_feature_selection = max_weighted_features
Ratios_Export_columns_with_feature_selection = np.array(Ratios_Export_columns_with_feature_selection)
Ratios_Export_columns_with_feature_selection = np.reshape(Ratios_Export_columns_with_feature_selection, (len(Ratios_Export_columns_with_feature_selection), 1))
Ratios_Export_columns_with_feature_selection = Ratios_Export_columns_with_feature_selection.T
print(Ratios_Export_columns_with_feature_selection)
Ratios_Export_with_feature_selection = Ratios_Export.iloc[:, feature_selection_column_indexes].values
concatenated = np.concatenate([Ratios_Export_columns_with_feature_selection, Ratios_Export_with_feature_selection])
print(concatenated)

#Store these in a CSV file
np.savetxt("Input_Data_with_Neo4j_Feature_Selection.csv", concatenated, delimiter=",", comments='', fmt='%s')


