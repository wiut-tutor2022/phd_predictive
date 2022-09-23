import csv
import numpy as np
import pandas as pd
import cypher_queries_uc1 as cq

dataset1 = pd.read_csv("Training_Data_with_Neo4j_Feature_Selection.csv")
dataset2 = pd.read_csv("Training_Output_2019_43.csv")


class NeuralNetwork():

    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)

        # converting weights to a 4 by 1 matrix with values [-1;1) (we need [-1;1])
        self.synaptic_weights =2* np.random.random((4, 1))-1

    def sigmoid(self, x):
        # the sigmoid function
        return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(self, x):
        # derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            output = self.think(training_inputs)

            # computing error rate for back-propagation
            error = training_outputs - output

            # performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        # passing the inputs via the neuron to get output
        # converting values to floats

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":
    # initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    # training data consisting of 16 input values and 1 output
    training_inputs = np.array(dataset1)
    print(training_inputs)
    training_outputs = np.array(dataset2).T
    print(training_outputs)

    # training taking place
    neural_network.train(training_inputs, training_outputs, 1500)

    # user_input_one = str(input("P1: "))
    # user_input_two = str(input("P2: "))
    # user_input_three = str(input("P3: "))
    # user_input_four = str(input("P4: "))
    # user_input_five = str(input("P5: "))
    # user_input_six = str(input("L1: "))
    # user_input_seven = str(input("L2: "))
    # user_input_eight = str(input("A1: "))
    # user_input_nine = str(input("A2: "))
    # user_input_ten = str(input("A3: "))
    # user_input_eleven = str(input("F1: "))
    # user_input_twelve = str(input("F2: "))
    # user_input_thirteen = str(input("F3: "))
    # user_input_fourteen = str(input("F4: "))
    # user_input_fourteen = str(input("SA1: "))
    # user_input_fourteen = str(input("SA2: "))

    with open('Input_Data_with_Neo4j_Feature_Selection.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ar_ratio = float(row[cq.ar_node_name])
            fs_ratio = float(row[cq.fs_node_name])
            lr_ratio = float(row[cq.lr_node_name])
            pr_ratio = float(row[cq.pr_node_name])

print("Company's Financial Ratios: ")
print("Chosen Activity Ratio: ", cq.ar_node_name, ar_ratio,)
print("Chosen Financial Sustainability Ratio: ", cq.fs_node_name, fs_ratio)
print("Chosen Liquidity Ratio: ", cq.lr_node_name, lr_ratio)
print("Chosen Profitability Ratio: ", cq.pr_node_name, pr_ratio)

print("The Result: ")
final_result = neural_network.think(np.array([ar_ratio, fs_ratio, lr_ratio, pr_ratio]))
final_result = np.round_(final_result, decimals=3, out=None)
print(final_result)

print("[1.] - the high risk of bankruptcy, [0.] - the company X is stable ")