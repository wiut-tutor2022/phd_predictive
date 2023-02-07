import csv
from datetime import datetime

import numpy as np
import pandas as pd

# Training Data
# df = pd.read_csv('Training_Data_2021_113.csv')
# df = pd.read_csv('2017_sem2_attendance_data3.csv')
df = pd.read_csv('2017_sem2_attendance_data5.csv')

df.fillna(0, inplace=True)
print(df.shape)
ratios_input = df.drop('joint', axis=1)
print(ratios_input.shape)
companies_output = df[['joint']].T
print(companies_output.shape)


class NeuralNetwork():

    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)

        # converting weights to a 14 by 1 matrix with values [-1;1) (we need [-1;1])
        # self.synaptic_weights = 2 * np.random.random((14, 1)) - 1 #updated by Olga
        # ValueError: shapes (2136,12) and (5,1) not aligned: 12 (dim 1) != 5 (dim 0)
        self.synaptic_weights = 2 * np.random.random((11, 1)) - 1

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
        inputs = inputs.astype(int)
        # inputs = inputs.astype(int or str or datetime)
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
    # neural_network.train(training_inputs, training_outputs, 1500) #updated by Olga below
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

    with open('Attributes_Export2.csv') as csvfile:  # Testing Company Input Data
        reader = csv.DictReader(csvfile)
        for row in reader:
            year = int(row['year'])
            semester = str(row['semester'])
            week = int(row['week'])
            # date = date(row['date'])
            day = str(row['day'])
            time_of_day = str(row['time_of_day'])
            #   start_time = datetime(row['start_time'])  # time
            #   end_time = datetime(row['end_time'])
            room_name = str(row['room_name'])
            class_type = str(row['class_type'])
            #   faculty = str(row['faculty'])
            #   school = str(row['school'])
            #joint = int(row['joint'])
            status = str(row['status'])
            degree = str(row['degree'])
            enrollment = int(row['enrollment'])
            #   class_duration = datetime(row['class_duration'])  # time variable
            attendance = int(row['attendance'])

print("Attributes predicting attendance")

print("Class type(lecture/laboratory/tutorial)*: ", class_type, )
#print("Course is combined or with other courses or not*", joint, )
# print("Faculty*: ", faculty, )
print("Enrollment*", enrollment, )
# print("School* ", school, )
print("Degree* ", degree, )
print("Status* (open or full) ", status, )
print("Room name: ", room_name, )
# print("Time of the day: ", time_of_day, )
# print("Day", day, )
print("Year", year, )
# print("Semester", semester, )
print("Week ", week, )
# print("Date", date, )
print("Enrollment", enrollment)
print("Attendance", attendance)

print("The Result: ")
# final_result = neural_network.think(np.array([year, semester, week, date, day,
#                                               time_of_day, start_time, end_time, room_name, class_type, faculty, school, joint
#                                               ])),

final_result = neural_network.think(np.array(
    [year, semester, week, day, time_of_day, room_name, class_type, status, degree, enrollment, attendance
     ])),

# ,year,semester,week,day,time_of_day,room_name,class_type,school,joint,status,degree,enrollment,attendance
# joint, status, degree, enrollment, class_duration, attendance
final_result = np.round(final_result, 5)
print(final_result)

print("[1.] - the high risk of bankruptcy, [0.] - the company X is stable ")
#chose joint to remove and as a point to rotate around, need to ask Natalya why