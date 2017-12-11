"""
FastMap.py
Implementation of the FastMap algorithm for Design and Analysis of Algorithms (CS 5110).
"""
import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn import preprocessing
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

__author__ = "Chris Campell & Patrick Beekman"
__version__ = "11/21/2017"

class FastMap:
    O = None
    X = None
    k = None
    PA = None
    col_num = None
    O_a = None
    a = None
    O_b = None
    old_D = None

    def __init__(self, k, O, D, col_num, PA, X):
        """

        :param k:
        :param O:
        :param D: The N by m pairwise distance matrix D.
        :param col_num:
        :param PA:
        :param X:
        """
        self.k = k
        self.O = O
        self.D = D
        self.col_num = col_num
        self.PA = PA
        self.X = X

    def object_coordinate(self, D, i):
        """
        object_coordinate: Helper method for d_prime; computes the first coordinate (x_i)	 of the provided object O_i.
        :param D:
        :param O_i: The object/row O_i for which the x coordinate of the object is to be determined prior to projection onto hyper-plane H.
        :param O_a: The first pivot object computed by choose_dist_objects.
        :param O_b: The second pivot object computed by choose_dist_objects.
        :return x_i: The object coordiante of object i in relation to the provided pivot objects.
        """
        x_i = ((self.old_D[(self.a, i)] ** 2) + (self.old_D[(self.a, self.b)] ** 2) - (self.old_D[(self.b, i)] ** 2)) \
              / (2 * self.old_D[(self.a, self.b)])
        # x_i = (((D(self.O_a, O_i) ** 2) + (D(self.O_a, self.O_b) ** 2)) - (D(self.O_b, O_i) ** 2)) / (2 * D(self.O_a, self.O_b))
        return x_i

    def d_prime(self, O):
        """
        d_prime: Computes the Eculidean distance D'() between objects O_i and O_j after projection onto the H hyper-plane.
        :param D: The original Euclidean distance of object O_i and O_j prior to projection onto hyper-plane H.
        :return d_prime: The Euclidean distance between objects O_i and O_j after projection onto the H hyper-plane.
        """
        d_prime = np.empty((len(O), len(O)))
        d_prime[:] = np.NaN
        for i, O_i in enumerate(O):
            for j, O_j in enumerate(O):
                x_i = self.object_coordinate(self.old_D, i)
                x_j = self.object_coordinate(self.old_D, j)
                d_prime[(i,j)] = np.sqrt(math.pow(self.old_D[(i, j)],2) - math.pow((x_i - x_j), 2))
        return d_prime


    def choose_dist_objects(self, O, D, alpha=5):
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
        # a != b:
        if b is not 0:
            a = 0
        else:
            a = 1
        O_a = O[a]
        # iterate through alpha times, getting better results each time
        for x in range(alpha):
            max_dist = 0
            # Get the object O_a farthest from O_b according to D(*,*)
            for i, O_i in enumerate(O):
                if i != b:
                    tmp = D[(b, i)]
                    if tmp > max_dist:
                        max_dist = tmp
                        O_a = O_i
                        a = i
            # Get the object O_b farthest from O_a according to D(*,*)
            max_dist = 0
            for i, O_i in enumerate(O):
                if i != a:
                    tmp = D[(a, i)]
                    if tmp > max_dist:
                        max_dist = tmp
                        O_b = O_i
                        b = i
        return O_a, a, O_b, b

    def fast_map(self, k, O, D, col_num):
        """
        fast_map: A fast algorithm which maps objects into points in a user defined k-dimensional space while preserving dis-similarities.
        :param k: The desired dimensionality of the output mapping.
        :param D:
        :param O: A Sample-by-Feature matrix of objects to transform into a k-dimensional space.
        """
        #col_num = 0
        if (k <= 0):
            return
        else:
            self.col_num += 1
        # Choose the pivot objects O_a and O_b:
        self.O_a, self.a, self.O_b, self.b = self.choose_dist_objects(O, D, 5)
        # Update the id's of the pivot objects:
        self.PA[0, self.col_num] = self.a
        self.PA[1, self.col_num] = self.b
        # TODO: Passing O_i=O_a incorrectly.
        if D[(self.a, self.b)] == 0:
            for i, row in enumerate(self.X):
                self.X[i, self.col_num] = 0
                # If the distance between row_a and row_b is zero
            return
        # Perform the projection onto line (O_a,O_b):
        for i, row in enumerate(O):
            x_i = self.object_coordinate(D, i)
            # Update the global array:
            self.X[i, self.col_num] = x_i
        self.old_D = D
        # Recurse:
        self.fast_map(k - 1, O, D=self.d_prime(O), col_num=col_num)

def main():
    # http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
    iris = load_wine()
    # Goal: Cluster different iris species using sepal length and width as features.
    # FastMap- Provide a distance function D(*,*). This distance function must be non-negative, symmetric, and obey the triangular inequality.
    # We will use the Euclidean distance L_{2} norm.
    # Use all the features in the Iris Dataset {sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)}:
    # O is an N * k array:
    O = iris.data[:, :]
    O = preprocessing.normalize(O)
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
    D = distance.squareform(distance.pdist(O, metric='euclidean'))
    # Call fast-map. Once this function is done executing the i-th row of global matrix X will be the image of the i-th row in the kth dimension:
    fm = FastMap(k=k,O=O,D=D,col_num=-1,PA=PA,X=X)
    fm.old_D = D
    fm.fast_map(k=k,O=O,D=D,col_num=-1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    for i, target in enumerate(y):
        if target == 0:
            xs = X[i,0]
            ys = X[i,1]
            zs = X[i,2]
            ax.scatter(xs, ys, zs, color='r', marker='o')
        elif target == 1:
            xs = X[i,0]
            ys = X[i,1]
            zs = X[i,2]
            ax.scatter(xs, ys, zs, color='b', marker='^')
        elif target == 2:
            xs = X[i,0]
            ys = X[i,1]
            zs = X[i,2]
            ax.scatter(xs, ys, zs, color='g', marker='*')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    # plt.clf()
    # for i, target in enumerate(y):
    #     if target == 0:
    #         xs = X[i,0]
    #         ys = X[i,1]
    #         # zs = O[i,2]
    #         # ax.scatter(xs, ys, zs, color='r', marker='o')
    #         plt.scatter(xs, ys, color='r', marker='o')
    #     elif target == 1:
    #         xs = X[i,0]
    #         ys = X[i,1]
    #         # zs = O[i,2]
    #         # ax.scatter(xs, ys, zs, color='b', marker='^')
    #         plt.scatter(xs, ys, color='b', marker='^')
    #     elif target == 2:
    #         xs = X[i,0]
    #         ys = X[i,1]
    #         # zs = O[i,2]
    #         # ax.scatter(xs, ys, zs, color='g', marker='*')
    #         plt.scatter(xs, ys, color='g', marker='*')
    # plt.show()

if __name__ == '__main__':
    main()
