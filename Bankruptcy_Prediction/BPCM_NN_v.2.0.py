import csv
import numpy as np
import pandas as pd

# Training Data
df = pd.read_csv('Training_Data_2021_113.csv')
df.fillna(0, inplace=True)
print(df.shape)
ratios_input = df.drop('Class',axis=1)
print(ratios_input.shape)
companies_output = df[['Class']].T
print(companies_output.shape)


class NeuralNetwork():

    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)

        # converting weights to a 14 by 1 matrix with values [-1;1) (we need [-1;1])
        self.synaptic_weights =2* np.random.random((14, 1))-1

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

    # print("Beginning Randomly Generated Weights: ")
    # print(neural_network.synaptic_weights)

    # training data consisting of 14 input values and 1 output
    training_inputs = np.array(ratios_input)
    # print(training_inputs)
    training_outputs = np.array(companies_output).T
    # print(training_outputs)

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

    with open('Ratios_Export.csv') as csvfile: # Testing Company Input Data
        reader = csv.DictReader(csvfile)
        for row in reader:
            return_on_equity            = float(row['Return_on_Equity'])
            return_on_capital_employed  = float(row['Return_on_Capital_Employed'])
            profit_margin               = float(row['Profit_Margin'])
            gross_margin                = float(row['Gross_Margin'])
            net_assets_turnover         = float(row['Net_Assets_Turnover'])
            current_ratio               = float(row['Current_Ratio'])
            liquidity_ratio             = float(row['Liquidity_Ratio'])
            creditors_payment           = float(row['Creditors_Payment'])
            debtors_turnover            = float(row['Debtors_Turnover'])
            stock_turnover              = float(row['Stock_Turnover'])
            gearing                     = float(row['Gearing'])
            interest_cover              = float(row['Interest_Cover'])
            cash_flow_coverage_ratio    = float(row['Cash_Flow_Coverage'])
            current_liability_coverage_ratio    = float(row['Current_Liability_Coverage'])


print("Company's Financial Ratios: ")

print("Return on Shareholders Funds: ", return_on_equity,)
print("Return on Capital Employed: ", return_on_capital_employed,)
print("Profit Margin: ", profit_margin,)
print("Gross Margin: ", gross_margin,)
print("Net Assets Turnover: ", net_assets_turnover,)
print("Current Ratio: ", current_ratio,)
print("Liquidity Ratio: ", liquidity_ratio,)
print("Creditors Payment: ", creditors_payment,)
print("Debtors Turnover: ", debtors_turnover,)
print("Stock Turnover: ", stock_turnover,)
print("Gearing: ", gearing,)
print("Interest Cover: ", interest_cover,)
print("Cash Flow Coverage Ratio: ", cash_flow_coverage_ratio,)
print("Current Liability Coverage Ratio: ", current_liability_coverage_ratio,)


print("The Result: ")
final_result = neural_network.think(np.array([return_on_equity, return_on_capital_employed, profit_margin, gross_margin,
          net_assets_turnover,  current_ratio, liquidity_ratio, creditors_payment, debtors_turnover, stock_turnover,
          gearing,  interest_cover, cash_flow_coverage_ratio, current_liability_coverage_ratio]))
final_result = np.round(final_result, 5)
print(final_result)

print("[1.] - the high risk of bankruptcy, [0.] - the company X is stable ")
