import pandas as p
import numpy as np


# To calculate slope
def beta_calculation(input_x, input_y):
    dot_transpose_arr = input_x.transpose().dot(input_x)
    input_x_inverse = np.linalg.pinv(dot_transpose_arr)
    input_x_transpose_y = input_x.transpose().dot(input_y)
    m = input_x_inverse.dot(input_x_transpose_y)
    return m


def calculate_y(input_x, beta):
    y = input_x.dot(beta)
    return y


# K- fold cross validation and calculation of error rates and accuracy
def k_fold(folds_count, input_data):
    error_rate = []
    input_x = input_data[['x0', 'x1', 'x2', 'x3', 'x4']]
    input_y = input_data[['x5']]
    input_y = input_y.replace({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3})
    folds_input = np.array_split(input_x, folds_count)
    folds_output = np.array_split(input_y, folds_count)
    for i in range(folds_count):
        testing_data_x = folds_input.pop(i)
        training_data_x = np.concatenate(folds_input)
        testing_data_y = folds_output.pop(i)
        training_data_y = np.concatenate(folds_output)
        beta = beta_calculation(training_data_x, training_data_y)
        y = calculate_y(testing_data_x, beta)
        residual = np.array(y) - np.array(testing_data_y)
        sum = np.sum((np.power(residual, 2)))
        error_rate.append(sum/len(testing_data_x))
        folds_input.insert(i, testing_data_x)
        folds_output.insert(i, testing_data_y)
        print("The error rate for fold {} is {}".format(i+1, error_rate[i]))
    print("The average error rate for {} folds is: {}".format(folds_count, np.sum(error_rate)/folds_count))
    print("The accuracy is: {0:.2f}%".format(100 - (np.sum(error_rate)/folds_count) * 100))
    #return np.sum(error_rate)/folds_count, 100 - (np.sum(error_rate)/folds_count) * 100, np.min(error_rate), error_rate.index(np.min(error_rate)) + 1


input_data = p.read_csv('iris.data', delimiter=',')
input_data.insert(0,'x0',1)
k_fold(15, input_data)