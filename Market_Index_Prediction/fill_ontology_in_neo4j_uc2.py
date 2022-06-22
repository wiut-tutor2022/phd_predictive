from neo4j import GraphDatabase
from owlready2 import *
import pandas as pd
import numpy as np

class OntotologyToGraphDatabase:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    # Establish connection neo4j db
    greeter = OntotologyToGraphDatabase("bolt://localhost:7687", "neo4j", "11111")
    # Processed input values
    processed_input = pd.read_csv("Processed_Input_Data_FTSE100_1985_21.csv", header=0).values
    # Weights values
    weights = pd.read_csv("weights_importance_uc2.csv", header=0).values
    weights = weights.T
    weights_columns = np.array(weights[0]).tolist()
    weights_values = np.array(weights[1]).tolist()
    # Load ontology
    onto_path.append(".")
    onto = get_ontology("Price_Index_Prediction_v3.2.owl")
    onto.load()
    # Get column names to be able to select values from csv
    processed_input_columns = pd.read_csv("Processed_Input_Data_FTSE100_1985_21.csv").columns.tolist()
    # Get first row for testing
    last_row = processed_input[-1]
    print(last_row)
    for row in processed_input:
        match_nodes_query = "MATCH "
        set_nodes_values_query = "SET "
        set_nodes_weights_query = "SET "
        complete_query = ""
        date = row[0]
        for c in onto.classes():
            if str(c.name) in processed_input_columns:
                match_nodes_query += "({cname}:Class {{name: '{cname}', Date: '{date}'}}), "\
                    .format(cname=c.name,
                            date=date)
                index_of_class = processed_input_columns.index(c.name)
                set_nodes_values_query += "{cname}.Value = {csv_value}, "\
                    .format(cname=c.name,
                           csv_value=row[index_of_class])
            if (str(c.name) in weights_columns) & (date == processed_input[-1][0]):
                index_of_weight = weights_columns.index(c.name)
                set_nodes_weights_query += "{cname}.Weight = {csv_weight}, " \
                    .format(cname=c.name,
                            csv_weight=weights_values[index_of_weight])
        match_nodes_query = match_nodes_query[:-2]
        set_nodes_values_query = set_nodes_values_query[:-2]
        if (date == processed_input[-1][0]):
            set_nodes_weights_query = set_nodes_weights_query[:-2]
        if (date == processed_input[-1][0]):
            complete_query = match_nodes_query + " " + set_nodes_values_query + " " + set_nodes_weights_query + " RETURN *"
        else:
            complete_query = match_nodes_query + " " + set_nodes_values_query + " RETURN *"
        print(match_nodes_query)
        print(set_nodes_values_query)
        print(set_nodes_weights_query)
        print(complete_query)
        with greeter.driver.session() as session:
            session.run(complete_query)

    greeter.close()