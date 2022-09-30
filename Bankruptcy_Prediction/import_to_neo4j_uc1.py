from neo4j import GraphDatabase
from owlready2 import *
import pandas as pd
import numpy as np

class neo4j_import:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def reset_graphs(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE (n)")

    def add_nodes_in_neo4j(self, ontology):
        add_nodes_query = ""
        IRI_prefix = "OBP."
        IRI_prefix_2 = "OBP_Ontology_v.2.1."

        def to_str(n):
            return str(n)

        data_properties = list(map(to_str, ontology.data_properties()))

        annotation_properties = set()
        for ap in ontology.annotation_properties():
            if ("rdf-schema.indexRelationships" not in str(ap)):
                annotation_properties.add(str(ap).replace(IRI_prefix, "").replace(IRI_prefix_2, ""))

        i = 0
        for c in ontology.classes():
            # First node part
            query = "CREATE (n%s:Class {" % i
            for p in data_properties:
                p = p.replace(IRI_prefix, "").replace(IRI_prefix_2, "")
                query += "%s: '', " % p
            query += "name: '%s'}) " % c.name.replace("'", "\\'")
            print(c.name)
            add_nodes_query += query
            i += 1
        print(add_nodes_query)
        add_nodes_query = add_nodes_query[:-1]
        with self.driver.session() as session:
            session.run(add_nodes_query)
        return add_nodes_query

    def add_annotation_edges_in_neo4j(self, ontology):
        IRI_prefix = "OBP."
        IRI_prefix_2 = "OBP_Ontology_v.2.1."
        annotation_properties = set()
        for ap in ontology.annotation_properties():
            if ("rdf-schema.indexRelationships" not in str(ap)):
                print(ap)
                annotation_properties.add(str(ap).replace(IRI_prefix, ""))
        i = 0
        match_nodes_query = "MATCH "
        create_edges_query = "CREATE "
        for c in ontology.classes():
            cname_formatted = c.name.replace(IRI_prefix_2, "").replace(".", "_").replace("-", "_").replace("'", "_").replace("(", "_").replace(")", "_").replace("&", "_")
            match_nodes_query += "({cname_formatted}:Class {{name: '{cname}'}}), "\
                .format(cname_formatted=cname_formatted,
                        cname=c.name.replace("'", "\\'"))
            print(match_nodes_query)
            # add relationships by annotation property
            for ap_unformatted in annotation_properties:
                ap = ap_unformatted.replace(IRI_prefix, "")
                print(ap)
                if hasattr(c, ap):
                    for rng in getattr(c, ap):
                        create_edges_query += "({cname_formatted})-[r{i}:{relname}]->({rng}), " \
                            .format(cname_formatted=cname_formatted,
                                    i=i,
                                    relname=ap,
                                    rng=str(rng).replace(IRI_prefix_2, "").replace(IRI_prefix, "").replace(".", "_").replace("-", "_").replace("'", "_").replace("(", "_").replace(")", "_").replace("&", "_"))
                        i += 1

            for supclass in c.is_a:
                if str(supclass).replace(IRI_prefix_2, "") != "owl.Thing":
                    create_edges_query += "({cname_formatted})-[r{i}:{relname}]->({supclass}), "\
                        .format(cname_formatted=cname_formatted,
                                i=i,
                                relname="subclass_of",
                                supclass=str(supclass).replace(IRI_prefix_2, "").replace(IRI_prefix, "").replace(".", "_").replace("-", "_").replace("'", "_").replace("(", "_").replace(")", "_").replace("&", "_"))
                i += 1
        match_nodes_query = match_nodes_query[:-2]
        create_edges_query = create_edges_query[:-2]
        print(match_nodes_query)
        print(create_edges_query)
        with self.driver.session() as session:
            session.run(match_nodes_query + " " + create_edges_query)
        return match_nodes_query, create_edges_query

if __name__ == "__main__":
    # DB parameters in Neo4j
    greeter = neo4j_import("bolt://localhost:7687", "neo4j", "000000")
    onto_path.append(".")
    onto = get_ontology("OBP_Ontology_v.2.1.owl")
    onto.load()
    greeter.reset_graphs()
    # Import ontology file into Neo4j
    greeter.add_nodes_in_neo4j(onto)
    greeter.add_annotation_edges_in_neo4j(onto)

    # Fill fin indicator nodes with data from csv
    company_fin_indicators = pd.read_csv("Company_A_Fin_Indicators.csv").values[0]
    company_fin_indicators_columns = pd.read_csv("Company_A_Fin_Indicators.csv").columns.tolist()
    company_fin_indicators_dict = {
        "Increase(Decrease) Cash & Equiv.\nth GBP Last avail. yr": "Operating_Cash_Flow",
        "Turnover\nth GBP Last avail. yr": "Sales",
        "Cost of Sales\nth GBP Last avail. yr": "Cost",
        "Gross Profit\nth GBP Last avail. yr": "Gross_Profit",
        "Interest Paid\nth GBP Last avail. yr": "Interest",
        "Profit (Loss) before Taxation\nth GBP Last avail. yr": "PBIT",
        "Trade Debtors\nth GBP Last avail. yr": "Trade_Debtors",
        "Stock & W.I.P.\nth GBP Last avail. yr": "Stock_&_W.I.P",
        "Current Assets\nth GBP Last avail. yr": "Current_Assets",
        "Total Assets\nth GBP Last avail. yr": "Total_Assets",
        "Total Assets less Cur. Liab.\nth GBP Last avail. yr": "Total_Assets_Less_Current_Liabilities",
        "Trade Creditors\nth GBP Last avail. yr": "Trade_Creditors",
        "Short Term Loans & Overdrafts\nth GBP Last avail. yr": "Short_Term_Loans_and_Overdrafts",
        "Current Liabilities\nth GBP Last avail. yr": "Current_Liabilities",
        "Long Term Liabilities\nth GBP Last avail. yr": "Longterm_Liabilities_(Debt)",
        "Shareholders Funds\nth GBP Last avail. yr": "Shareholders'_Funds_(Equity)"
    }

    match_nodes_query = "MATCH "
    IRI_prefix_2 = "OBP_Ontology_v.2.1."
    for c in onto.classes():
        cname_formatted = c.name\
            .replace(IRI_prefix_2, "")\
            .replace(".", "_")\
            .replace("-", "_")\
            .replace("'", "_")\
            .replace("(", "_")\
            .replace(")", "_")\
            .replace("&", "_")
        match_nodes_query += "({cname_formatted}:Class {{name: '{cname}'}}), " \
            .format(cname_formatted=cname_formatted,
                    cname=c.name.replace("'", "\\'"))
    set_values_query = "SET "
    i = 0
    for col_name in company_fin_indicators_columns:
        matching_node_name = company_fin_indicators_dict[col_name]
        matching_node_name = matching_node_name\
            .replace(".", "_")\
            .replace("-", "_")\
            .replace("'", "_")\
            .replace("(", "_")\
            .replace(")", "_")\
            .replace("&", "_")
        set_values_query += f"{matching_node_name}.Value = {company_fin_indicators[i]}, "
        i += 1

    match_nodes_query = match_nodes_query[:-2]
    set_values_query = set_values_query[:-2]

    with greeter.driver.session() as session:
        session.run(match_nodes_query + " " + set_values_query)

    # Calculate and fill fin ratio nodes
    ratio_calcs_dict = {
        "Return_on_Equity": "(PBIT.Value / Shareholders__Funds__Equity_.Value) * 100",
        "Return_on_Capital_Employed": "(PBIT.Value / Total_Assets_Less_Current_Liabilities.Value) * 100",
        "Profit_Margin": "(PBIT.Value / Sales.Value) * 100",
        "Gross_Margin": "(Gross_Profit.Value / Sales.Value) * 100",
        "Net_Assets_Turnover": "Sales.Value / Total_Assets_Less_Current_Liabilities.Value",
        "Current_Ratio": "-Current_Assets.Value / Current_Liabilities.Value",
        "Liquidity_Ratio": "(Stock___W_I_P.Value - Current_Assets.Value) / Current_Liabilities.Value",
        "Creditors_Payment": "Cost.Value / Trade_Creditors.Value",
        "Debtors_Turnover": "Sales.Value / Trade_Debtors.Value",
        "Stock_Turnover": "Sales.Value / Stock___W_I_P.Value",
        "Gearing": "(-(Short_Term_Loans_and_Overdrafts.Value + Longterm_Liabilities__Debt_.Value) / Shareholders__Funds__Equity_.Value) * 100",
        "Interest_Cover": "-PBIT.Value / Interest.Value",
        "Cash_Flow_Coverage": "-Operating_Cash_Flow.Value / Longterm_Liabilities__Debt_.Value",
        "Current_Liability_Coverage": "-Operating_Cash_Flow.Value / Current_Liabilities.Value",
    }

    set_ratios_query = "SET "
    for c in onto.classes():
        cname_formatted = c.name\
            .replace(IRI_prefix_2, "")\
            .replace(".", "_")\
            .replace("-", "_")\
            .replace("'", "_")\
            .replace("(", "_")\
            .replace(")", "_")\
            .replace("&", "_")
        if c.name in ratio_calcs_dict:
            set_ratios_query += f"{cname_formatted}.Value = {ratio_calcs_dict[c.name]}, "
    set_ratios_query = set_ratios_query[:-2]
    print(set_ratios_query)

    with greeter.driver.session() as session:
        session.run(match_nodes_query + " " + set_ratios_query)

    # Add weights values to nodes
    weights = pd.read_csv("weights_importance_uc1.csv", header=0).values
    weights = weights.T
    weights_columns = np.array(weights[0]).tolist()
    weights = weights[1]
    set_weights_query = "SET "
    for c in onto.classes():
        if str(c.name) in weights_columns:
            index_for_weight = weights_columns.index(c.name)
            set_weights_query += f"{c.name}.Weight = {weights[index_for_weight]}, "
    # match_nodes_query = match_nodes_query[:-2]
    set_weights_query = set_weights_query[:-2]
    print(set_weights_query)

    with greeter.driver.session() as session:
        session.run(match_nodes_query + " " + set_weights_query)

    fin_ratios = [
        "Return_on_Equity",
        "Return_on_Capital_Employed",
        "Profit_Margin",
        "Gross_Margin",
        "Net_Assets_Turnover",
        "Current_Ratio",
        "Liquidity_Ratio",
        "Creditors_Payment",
        "Debtors_Turnover",
        "Stock_Turnover",
        "Gearing",
        "Interest_Cover",
        "Cash_Flow_Coverage",
        "Current_Liability_Coverage",
    ]
    fin_ratios = np.array(fin_ratios)
    result = ""
    with greeter.driver.session() as session:
        result = session.run("MATCH (n) RETURN (n)").data()
    nodes = []
    for n in result:
        n_props = n["n"]
        nodes.append(n_props)
    nodes = np.array(nodes).tolist()
    ratio_values = []
    i = 1
    for ratio in fin_ratios:
        for n in nodes:
            if n["name"] == ratio:
                print(n["name"])
                print(n["Value"])
                print(i)
                ratio_values.append(n["Value"])
                i += 1
    ratio_values = np.array(ratio_values)
    fin_ratios = np.reshape(fin_ratios, (len(fin_ratios), 1))
    ratio_values = np.reshape(ratio_values, (len(ratio_values), 1))

    concatenated = np.concatenate([fin_ratios, ratio_values], axis=1)
    concatenated = concatenated.T
    print(concatenated)
    np.savetxt("Ratios_Export.csv", concatenated, delimiter=",", comments='', fmt='%s')

    greeter.close()
