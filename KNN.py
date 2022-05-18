import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification


def min_max_normalize(lst):
    """Funkcja do normalizowania danych

    Args:
        lst (list): Lista punktów

    Returns:
        list: lista danych znormalizowana
    """
    minimum = min(lst)
    maximum = max(lst)
    return [(x - minimum) / (maximum - minimum) for x in lst]


def euclidean_distance(element1, element2):
    """Funkcja do obliczania odleglosci euklidesowej

    Args:
        element1 (list): pierwszy punkt 2D
        element2 (list): drugi punkt 2D

    Returns:
        float: odleglosc euklidesowa danych punktów
    """
    x_distance = (element1[0] - element2[0])**2
    y_distance = (element1[1] - element2[1])**2
    return (x_distance + y_distance)**0.5


def get_label(neighbours, y):
    """Funkcja do sprawdzenia klas dla najbliższych sąsiadów i na podstawie tej klasy, ustawienie klasy dla punktu
    testowego

    Args:
        neighbours (list): lista k najbliższych sąsiadów
        y (list): lista klas dla punktów zbioru treningowego

    Returns:
        int: klasa dla punktu testowego
    """
    zero_count, one_count = 0, 0
    for element in neighbours:
        if y[element[1]] == 0:
            zero_count += 1
        elif y[element[1]] == 1:
            one_count += 1
    if zero_count == one_count:
        return y[neighbours[0][1]]
    return 1 if one_count > zero_count else 0


def find_nearest(x, y, _input, k):
    """Funkcja do znalezienia wszystkich odległości między punktem testowym a punktami zbioru treningowego. Następnie
    funkcja wywołuje funkcję get_label() i znajduję klasę dla punktu testowego

    Args:
        x (list): Zbiór treningowy zawierający dwie klasy
        y (list):
        _input: Lista zawierająca klasy punktów, które chcemy sklasyfikować
        k (int): Liczba najbliższych sąsiadów

    Returns:
        int: klasa dla punktu testowego
        list: k najbliższych sąsiadów/punktów
        list:
    """
    distances = []
    for id, element in enumerate(x):
        distances.append([euclidean_distance(_input, element), id])
    distances = sorted(distances)
    predicted_label = get_label(distances[0:k], y)
    return predicted_label, distances[0:k], distances[k:]


# x, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=2)
x, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=3
    )


# Normalizacja danych
x[:, 0] = min_max_normalize(x[:, 0])
x[:, 1] = min_max_normalize(x[:, 1])

# Transformacja danych do dataframe
df = pd.DataFrame(x, columns=['Feature1', 'Feature2'])
df['Label'] = y
st.dataframe(df)

# Inicjalizacja streamlit
st.title('KNN')
x_input = st.slider("Wybierz współrzędną x punktu testowego", min_value=0.0, max_value=1.0, key='x')
y_input = st.slider("Wybierz współrzędną y punktu testowego", min_value=0.0, max_value=1.0, key='y')
k = st.slider("Wybierz liczbę najbliższych sąsiadów", min_value=1, max_value=10, key='k')
_input = (x_input, y_input)

# Wykres punktów
fig = px.scatter(df, x='Feature1', y='Feature2', symbol='Label', symbol_map={0: 'square-dot', 1: 'circle'})
fig.add_trace(go.Scatter(x=[_input[0]], y=[_input[1]], name="Punkt do klasyfikacji",))
fig.add_trace(go.Contour(z=y, x=df['Feature1'], y=df['Feature2'], colorscale='Rdbu', showscale=False))
st.plotly_chart(fig)

# Szukanie najbliższych sąsiadów
predicted_label, nearest_neighbours, far_neighbours = find_nearest(x, y, _input, k)
st.title('Predykcja')
st.subheader('Predicted label: {}'.format(predicted_label))
nearest_neighbours = [[neighbour[1], x[neighbour[1], 0], x[neighbour[1], 1], neighbour[0], y[neighbour[1]]]
                      for neighbour in nearest_neighbours]

nearest_neighbours = pd.DataFrame(nearest_neighbours, columns=['Index', 'Feature1', 'Feature2', 'Distance', 'Label'])
st.dataframe(nearest_neighbours)

# Szukanie najdalszych sąsiadów
far_neighbours = [[neighbour[1], x[neighbour[1], 0], x[neighbour[1], 1], neighbour[0], y[neighbour[1]]]
                  for neighbour in far_neighbours]
