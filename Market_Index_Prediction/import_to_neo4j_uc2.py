from neo4j import GraphDatabase
from owlready2 import *
import pandas as pd

class OntotologyToGraphDatabase:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def print_greeting(self, message):
        with self.driver.session() as session:
            greeting = session.write_transaction(self._create_and_return_greeting, message)
            print(greeting)

    @staticmethod
    def _create_and_return_greeting(tx, message):
        result = tx.run("CREATE (a:Greeting) "
                        "SET a.message = $message "
                        "RETURN a.message + ', from node ' + id(a)", message=message)
        return result.single()[0]

    def reset_graphs(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE (n)")

    def add_nodes_in_neo4j(self, ontology, date, row, add_nodes_query):
        print(date)
        IRI_prefix = "PriceIndexPrediction."

        def to_str(n):
            return str(n)

        data_properties = list(map(to_str, ontology.data_properties()))

        annotation_properties = set()
        for ap in ontology.annotation_properties():
            if ("rdf-schema.indexRelationships" not in str(ap)):
                annotation_properties.add(str(ap).replace(IRI_prefix, ""))

        i = 0
        for c in ontology.classes():
            # First node part
            query = "CREATE (a%s%s:Class {" % (row, i)
            for p in data_properties:
                p = p.replace(IRI_prefix, "")
                if (p == "Date"):
                    query += "%s: '%s'" % (p, date)
                else:
                    query += "%s: ''" % p
                query += ", "
            c_str = str(c).replace(IRI_prefix, "")
            query += "name: '%s'}) " % c_str
            add_nodes_query += query
            i += 1
        return add_nodes_query

    def add_annotation_edges_in_neo4j(self, ontology, date, row):
        print(date)
        IRI_prefix = "PriceIndexPrediction."
        annotation_properties = set()
        for ap in ontology.annotation_properties():
            if ("rdf-schema.indexRelationships" not in str(ap)):
                annotation_properties.add(str(ap).replace(IRI_prefix, ""))
        i = 0
        match_nodes_queries = ""
        create_edges_queries = ""
        for c in ontology.classes():
            # add relationships by annotation property
            for ap in annotation_properties:
                if hasattr(c, ap):
                    for rng in getattr(c, ap):
                        match_nodes_queries = "MATCH (a{row}{i}:Class {{name: '{cname}', Date: '{date}'}}), (b{row}{i}:Class {{name: '{rng}', Date: '{date}'}})"\
                            .format(row=row,
                                    i=i,
                                    cname=c.name,
                                    date=date,
                                    rng=str(rng).replace(IRI_prefix, ""))
                        create_edges_queries = "CREATE (a{row}{i})-[r{row}{i}:{relname}]->(b{row}{i})" \
                            .format(row=row,
                                    i=i,
                                    relname=ap)
                        with self.driver.session() as session:
                            session.run(match_nodes_queries + " " + create_edges_queries)
                        i += 1
        return match_nodes_queries, create_edges_queries

    def add_subclass_edges_in_neo4j(self, ontology, date, row):
        print(date)
        IRI_prefix = "PriceIndexPrediction."
        i = 0
        match_nodes_queries = ""
        create_edges_queries = ""
        for c in ontology.classes():
            for supclass in c.is_a:
                match_nodes_queries = "MATCH (a{row}{i}:Class {{name: '{cname}', Date: '{date}'}}), (b{row}{i}:Class {{name: '{supclass}', Date: '{date}'}})"\
                    .format(row=row,
                            i=i,
                            cname=c.name,
                            date=date,
                            supclass=str(supclass).replace(IRI_prefix, ""))
                create_edges_queries = "CREATE (a{row}{i})-[r{row}{i}:{relname}]->(b{row}{i})"\
                .format(row=row,
                        i=i,
                        relname="subclass_of")
                with self.driver.session() as session:
                    session.run(match_nodes_queries + " " + create_edges_queries)
                i += 1
        return match_nodes_queries, create_edges_queries

if __name__ == "__main__":
    # DB password is last parameter of OntotologyToGraphDatabase() method (11111)
    greeter = OntotologyToGraphDatabase("bolt://localhost:7687", "neo4j", "11111")
    onto_path.append(".")
    onto = get_ontology("Price_Index_Prediction_v3.2.owl")
    onto.load()
    processed_input_csv_dates = pd.read_csv("Processed_Input_Data_FTSE100_1985_21.csv").values[:, 0]
    greeter.reset_graphs()
    add_nodes_query = ""
    match_nodes_queries = ""
    create_edges_queries = ""
    i = 0
    row = 0
    for date in processed_input_csv_dates:
        add_nodes_query = greeter.add_nodes_in_neo4j(onto, date, row, add_nodes_query)
        row += 1
        i += 1
        if (i % 10 == 0):
            with greeter.driver.session() as session:
                session.run(add_nodes_query)
            add_nodes_query = ""
    i = 0
    row = 0
    for date in processed_input_csv_dates:
        greeter.add_annotation_edges_in_neo4j(onto, date, row)
        greeter.add_subclass_edges_in_neo4j(onto, date, row)
        row += 1
        i += 1

    greeter.close()