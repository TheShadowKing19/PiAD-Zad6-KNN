import math as m
import sklearn.datasets as ds
import numpy as np


class knn:
    X, y = ds.make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=3
    )

    def __init__(self, n_neighbors=1, use_KDTree=False):
        self.n_neighbors = n_neighbors
        self.use_KDTree = use_KDTree

    def fit(self, X, y):
        """ Dopasuj punkt zbioru X do zbioru y """
        pass

    def predict(self, X):
        """ Sprawdza który z klas zbioru y jest najbliższy zbiorowi X """
        test_point = X[0]
        distances = []
        np.delete(X, 0, 0)
        for point in range(self.n_neighbors):



        pass

    def score(self, X, y):
        pass


if __name__ == '__main__':
    KNN = knn()
    row0 = KNN.X[0]
    print(KNN.X, KNN.y, row0, sep='\n')
    print(np.delete(KNN.X, 0, 0))


