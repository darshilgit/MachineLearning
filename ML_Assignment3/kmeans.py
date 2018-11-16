import pandas as p
import numpy as np
from matplotlib import pyplot as plt


input_data = p.read_csv('iris.data', delimiter=',')
# input_data.insert(0,'x0',1)
input_x = input_data[['x0', 'x1', 'x2', 'x3']]
# input_y = input_data[['x4']]
# input_y = input_y.replace({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3})

point_to_centroid = [-1]*150
count = 0


def calculate_centroid(k):
    centroids = input_x.sample(k)
    centroids.index = range(k)
    return centroids


def calculate_distance(centroids):
    for i in range(len(input_x)):
        point = input_x.loc[i]
        centroid_index = -1
        min_distance = 1000
        for j in range(len(centroids)):
            difference = centroids.values[j] - point.values
            difference = np.square(difference)
            distance = np.sum(difference)
            if min_distance > distance:
                min_distance = distance
                centroid_index = j
        point_to_centroid[i] = centroid_index


def update_centroid(point_to_centroid, input_x, centroids):
    input_x = input_x.values
    for i in range(len(centroids)):
        sum = np.zeros(len(centroids.values[0]))
        count = 0
        for j in range(len(input_x)):
            if i == point_to_centroid[j]:
                count += 1
                sum = sum + input_x[j]
        centroids.loc[i:i, :] = sum/count
    return centroids


def k_means(centroids, iter):
    while iter > 0:
        calculate_distance(centroids)
        centroids = update_centroid(point_to_centroid, input_x, centroids)
        iter -= 1
    return centroids


# https://www.linkedin.com/pulse/finding-optimal-number-clusters-k-means-through-elbow-asanka-perera/
# The below method uses the elbow method to find out the optimal number of clusters.
# The graph shows that the number of clusters should be 3.
def wcss(centroids, point_to_centroid):
    squared_difference = 0.0
    for i in range(len(input_x)):
        point = input_x.loc[i]
        centroid_of_class_of_point = centroids.loc[point_to_centroid[i]]
        difference = centroid_of_class_of_point.values - point.values
        squared_difference += np.sum(np.square(difference))
    return (squared_difference)

plot_wss = []


# for K = 3, the range will be (1,4), so that the plot starts from K = 1
def perform_kmeans(k):
    for i in range(1, k+1):
        centroids = calculate_centroid(i)
        centroids = k_means(centroids, 5)
        plot_wss.append(wcss(centroids, point_to_centroid))

    plt.plot(range(1, k+1), plot_wss, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Within Cluster Sum Of Errors')
    plt.show()


perform_kmeans(3)
