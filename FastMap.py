"""
FastMap.py
Implementation of the FastMap algorithm for Design and Analysis of Algorithms (CS 5110).
"""
import math
import numpy as np
from sklearn.datasets import load_iris
# from scipy.spatial.distance import euclidean

__author__ = "Chris Campell & Patrick Beekman"
__version__ = "11/21/2017"

O = None
X = None
k = None
PA = None
col_num = None
O_a = None
O_b = None



def object_coordinate(D, O_i, O_a, O_b):
    """
    object_coordinate: Helper method for d_prime; computes the first coordinate (x_i)	 of the provided object O_i.
    :param D: A user-provided (preferribly domain-expert) distance function.
    :param O_i: The object/row O_i for which the x coordinate of the object is to be determined prior to projection onto hyper-plane H.
    :param O_a: The first pivot object computed by choose_dist_objects.
    :param O_b: The second pivot object computed by choose_dist_objects.
    :return x_i: The object coordiante of object i in relation to the provided pivot objects.
    """
    x_i = (((D(O_a, O_i) ** 2) + (D(O_a, O_b) ** 2)) - (D(O_b, O_i) ** 2)) / (2 * D(O_a, O_b))
    return x_i


def euclidean(O_i, O_j):
    """
    euclidean: Returns the euclidean distance between objects O_i, O_j
    :param O_i: The first object
    :param O_j: The second object
    :return:
    """
    euclidean_dist = np.sqrt(np.sum([(x_1 - x_2)**2 for (x_1,x_2) in zip(O_i,O_j)]))
    return euclidean_dist


def d_prime(O_i, O_j, D=euclidean):
    """
    d_prime: Computes the Eculidean distance D'() between objects O_i and O_j after projection onto the H hyper-plane.
    :param D: The original Euclidean distance of object O_i and O_j prior to projection onto hyper-plane H.
    :param O_i: Object O_i prior to projection onto the hyper-plane H.
    :param O_j: Object O_j prior to projection onto the hyper-plane H.
    :return d_prime: The Euclidean distance between objects O_i and O_j after projection onto the H hyper-plane.
    """
    x_i = object_coordinate(D, O_i, O_a, O_b)
    x_j = object_coordinate(D, O_j, O_a, O_b)
    d_prime = np.sqrt(math.pow(D(O_i, O_j),2) - math.pow((x_i - x_j), 2))
    return d_prime


def choose_dist_objects(O, D, alpha=5):
    """
    choose_dist_objects: Chooses a pair of objects that are the furthest Euclidean distance from each other
    :param O: A Sample-by-Feature matrix of objects to transform into a k-dimensional space.
    :param D: A user-provided (preferribly domain-expert) distance function.
    :param alpha: A constant specifying the number of times to run this heuristic function.
    :return Oa: The object O_a who is furthest away from Ob
    :return a: The row index of object O_a.
    :return Ob: The object O_b who is furthest away from Oa
    :return b: The row index of object O_b.
    """
    b = np.random.randint(0, len(O) - 1)
    O_b = O[b]
    a = None
    O_a = None
    # iterate through alpha times, getting better results each time
    for x in range(alpha):
        max_dist = 0
        # Get the object O_a farthest from O_b according to D(*,*)
        for i, O_i in enumerate(O):
            if i != b:
                tmp = D(O_b, O_i)
                if tmp > max_dist:
                    max_dist = tmp
                    O_a = O_i
                    a = i
        # Get the object O_b farthest from O_a according to D(*,*)
        max_dist = 0
        for i, O_i in enumerate(O):
            if i != a:
                tmp = D(O_a, O_i)
                if tmp > max_dist:
                    max_dist = tmp
                    O_b = O_i
                    b = i
    return O_a, a, O_b, b

def fast_map(k, O, D):
    """
    fast_map: A fast algorithm which maps objects into points in a user defined k-dimensional space while preserving dis-similarities.
    :param k: The desired dimensionality of the output mapping.
    :param D: A user-provided (preferribly domain-expert) distance function.
    :param O: A Sample-by-Feature matrix of objects to transform into a k-dimensional space.
    """
    col_num = 0
    if (k <= 0):
        return
    else:
        col_num += 1
    # Choose the pivot objects O_a and O_b:
    O_a, a, O_b, b = choose_dist_objects(O, D, 5)
    # Update the id's of the pivot objects:
    PA[0, col_num] = a
    PA[1, col_num] = b
    if (euclidean(O_a, O_b) == 0):
        for i, row in enumerate(X):
            X[i, col_num] = 0
            # If the distance between row_a and row_b is zero
        return
    # Perform the projection onto line (O_a,O_b):
    for i, row in enumerate(X):
        x_i = object_coordinate(D, row, O_a, O_b)
        # Update the global array:
        X[i, col_num] = x_i
    # Recurse:
    fast_map(k - 1, O, D=d_prime)

def main():
    # Call fast-map. Once this function is done executing the i-th row of global matrix X will be the image of the i-th row in the kth dimension:
    fast_map(k=k, O=O, D=euclidean)

if __name__ == '__main__':
    # http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
    iris = load_iris()
    # Goal: Cluster different iris species using sepal length and width as features.
    # FastMap- Provide a distance function D(*,*). This distance function must be non-negative, symmetric, and obey the triangular inequality.
    # We will use the Euclidean distance L_{2} norm.
    # Use all the features in the Iris Dataset {sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)}:
    # O is an N * k array:
    O = iris.data[:, :]
    y = iris.target  # 0, 1, 2
    # Create a column pointer which references the column of the X array currently being updated.
    col_num = 0
    # Define the desired dimensionality of the output (k):
    k = 3
    # A 2 by k pivot array PA; stores the ids of the pivot objects- one pair per recursive call.
    PA = np.zeros((2, k))
    # Define an array X to hold the result of FastMap:
    X = np.empty((len(O), k))
    X[:] = np.NaN
    main()