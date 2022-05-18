import math as m
import sklearn.datasets as ds
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class knn:
    def __init__(self, n_neighbors=1, use_KDTree=False):
        self.n_neighbors = n_neighbors
        self.use_KDTree = use_KDTree
        self.nearest_neighbors_classes = None
        self.X = None
        self.y = None


    def fit(self, X, y):
        """
        Ustawia dane treningowe.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Zbiór punktów.
        y : array-like, shape (n_samples)
            Klasy punktów.
        """
        self.X = X
        self.y = y

    def predict(self, X):
        """
        Przewiduje klasę dla podanych danych testowych.

        Parameters
        ----------
        X : array-like
        method : str, possible values: 'majority', 'probability'
            Metoda dokonująca predykcji.

        Returns
        -------
        prediction : array-like, shape (n)
            Przewidziana klasa lub klasy, jeśli wiele najbliższych punktów
        """
        distances = []
        for i in range(len(X)):
            temp = []
            for indeks in range(len(self.X)):
                dist = m.dist(X[i], self.X[indeks])
                temp.append([dist, indeks])
            distances.append(temp)
        for i in range(len(distances)):
            distances[i].sort(key=lambda x: x[0])

        result = []
        for i in range(len(distances)):
            temp = []
            for j in range(self.n_neighbors):
                temp.append(self.y[distances[i][j][1]])
            result.append(temp)

        for i in range(len(result)):
            val, count = np.unique(result[i], return_counts=True)
            result[i] = val[np.argmax(count)]
        return result


    def score(self, X, y):
        pass

    def _set_nearest_neighbours_classes(self, nearest_neighbors_classes):
        """ Ustawia jakie klasy są najbliżej test_data
            Funkcja prywatna, używana przez fit()
            Funkcje prywatne nie są importowane za pomocą import
        """
        self.nearest_neighbors_classes = nearest_neighbors_classes




if __name__ == '__main__':
    X, y = ds.make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=3
    )
    test_data, _ = ds.make_classification(
        n_samples=2,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=2
    )
    knn_classifier = knn(n_neighbors=4, use_KDTree=False)
    knn_classifier.fit(X, y)
    prediction = knn_classifier.predict(test_data)
    max_x = max(X[:, 0])
    min_x = min(X[:, 0])
    max_y = max(X[:, 1])
    min_y = min(X[:, 1])
    x = np.linspace(min_x, max_x, 100)
    y = np.linspace(min_y, max_y, 100)
    meshx, meshy = np.meshgrid(x, y, sparse=False, indexing='xy')
    knn_classifier.fit(X, y)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    Z = knn_classifier.predict(np.c_[meshx.ravel(), meshy.ravel()])
    Z = np.array(Z).reshape(meshx.shape)
    plt.pcolormesh(meshx, meshy, Z, cmap=cmap_light)
    plt.contour(meshx, meshy, Z, cmap=cmap_bold)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(meshx.min(), meshx.max())
    plt.ylim(meshy.min(), meshy.max())
    plt.show()












