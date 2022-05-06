import numpy as np
import pandas as pd
import scipy.stats

url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/abalone/abalone.data"
)
abalone = pd.read_csv(url, header=None)
abalone.columns = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
    "Rings",
]
abalone = abalone.drop("Sex", axis=1)
print(abalone.head())
X = abalone.drop("Rings", axis=1)
print(X.head())
X = X.values
y = abalone["Rings"]
y = y.values
print(X, y, sep="\n")
new_data_point = np.array(  # To jest nowy punkt, której predykcji chcemy dokonać
    [0.569552, 0.446407, 0.154437, 1.016849, 0.439051, 0.222526, 0.291208, ]
)

distances = np.linalg.norm(X - new_data_point, axis=1)  # Obliczamy odległości nowego punktu od wszystkich innych, które
                                                        # już istniały
print(distances)
k = 3 # Wybieramy k-najbliższych sąsiadów
nearest_neighbors_indexes = np.argsort(distances)[:k]  # Sortujemy dystanse od najmniejszej do największej i pobieramy
# indeksy
print(nearest_neighbors_indexes)

nearest_neighbors_rings = y[nearest_neighbors_indexes]  # Sprawdzamy klasy k najbliższych punktów do naszego nowego
# punktu
print(nearest_neighbors_rings)

'''
 Regresja. Klasy są wartościami liczbowymi, dlatego wyciągnięcie średniej z klas najbliższych k-sąsiadów,
 pozwoli nam przewidzieć klase nowego punktu'''
prediction = nearest_neighbors_rings.mean()
print(prediction)   # Rozwiązanie zadania


# Klasyfikacja. Klasy są kategoriami (stringami). Wtedy używamy scipy.stats.mode(class_neighbors)
class_neighbors = np.array(["A", "B", "B", "C"])
rozw, razy = scipy.stats.mode(class_neighbors)
'''
scipy.stats.mode(class_neighbors) zwraca dwie wartości:
1. rozw - to najczęściej występująca klasa. Jeśli x klas pojawia się tyle samo razy, to zwraca tabele tych klas
2. razy - to liczba wystąpień tych klas, też jako tablica numpy
'''
print(rozw)
