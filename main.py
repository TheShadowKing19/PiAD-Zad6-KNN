import math as m
import sklearn.datasets as ds


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

    def fit(self, X, y):    # Obliczanie odległości między punktami. Y to pojedyńczy punkt.
        odl = 0.0
        for i in range(len(X)-1):
            odl += (X[i] - y[i])**2
        return m.sqrt(odl)

    def predict(self, X):
        pass

    def score(self, X, y):
        pass


if __name__ == '__main__':
    KNN = knn()
    row0 = KNN.X[0]

