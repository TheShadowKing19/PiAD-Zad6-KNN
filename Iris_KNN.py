import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import KNN


def _get_label_iris(neighbours, y):
    """Funkcja do sprawdzenia klas dla najbliższych sąsiadów i na podstawie tej klasy, ustawienie klasy dla punktu
    testowego

    Args:
        neighbours (list): lista k najbliższych sąsiadów
        y (list): lista klas dla punktów zbioru treningowego

    Returns:
        int: klasa dla punktu testowego
    """
    zero_count, one_count, two_count = 0, 0, 0
    for element in neighbours:
        if y[element[1]] == 0:
            zero_count += 1
        elif y[element[1]] == 1:
            one_count += 1
        elif y[element[1]] == 2:
            two_count += 1
    if zero_count > one_count and zero_count > two_count:
        return 0
    elif one_count > zero_count and one_count > two_count:
        return 1
    elif two_count > zero_count and two_count > one_count:
        return 2


def find_nearest_iris(x, y, _input, k):
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
        list: k najdalszych sąsiadów/punktów
    """
    distances = []
    for id, element in enumerate(x):
        distances.append([KNN.euclidean_distance(_input, element), id])
    distances = sorted(distances)
    predicted_label = _get_label_iris(distances[0:k], y)
    return predicted_label, distances[0:k], distances[k:]


ds = load_iris()

# Przygotowanie danych i normalizacja danych
iris = pd.read_csv('iris.csv')
x_iris = iris.iloc[:, 0:4].values   # Wyodrębniam kolumny z cechami irysów. To będzie x
y_iris = ds.target    # Pobieram klasy z datasetu sklearn
x_iris[:, 0] = KNN.min_max_normalize(list(x_iris[:, 0]))
x_iris[:, 1] = KNN.min_max_normalize(list(x_iris[:, 1]))
x_iris[:, 2] = KNN.min_max_normalize(list(x_iris[:, 2]))
x_iris[:, 3] = KNN.min_max_normalize(list(x_iris[:, 3]))
st.title('Iris classification')
st.subheader('Znormalizowane dane')

# PCA
pca = PCA(n_components=2)
PCA_iris = pca.fit_transform(x_iris)
PCA_df = pd.DataFrame(data=PCA_iris, columns=['PCA1', 'PCA2'])
PCA_df['variety'] = y_iris
st.dataframe(PCA_df)

# Input punkty testowego
x_input_iris = st.slider("Wybierz współrzędną x punktu testowego",
                         min_value=round(min(PCA_df['PCA1']), 2),
                         max_value=round(max(PCA_df['PCA1']), 2),
                         key='x_iris')
y_input_iris = st.slider("Wybierz współrzędną y punktu testowego",
                         min_value=round(min(PCA_df['PCA2']), 2),
                         max_value=round(max(PCA_df['PCA2']), 2),
                         key='y_iris')
k_iris = st.slider("Wybierz liczbę najbliższych sąsiadów", min_value=1, max_value=10, key='k_iris')
_input = (x_input_iris, y_input_iris)

# Wykres
fig = px.scatter(PCA_df, x='PCA1', y='PCA2', symbol='variety', symbol_map={0: 'circle', 1: 'square', 2: 'triangle-up'})
fig.add_trace(go.Scatter(x=[_input[0]], y=[_input[1]], name='Punkt do klasyfikacji'))
fig.add_trace(go.Contour(z=y_iris, x=PCA_df['PCA1'], y=PCA_df['PCA2'], colorscale='Rdbu', showscale=False))
st.plotly_chart(fig)

# Szukanie najbliższych sąsiadów
iris_predicted_label, iris_nearest_neighbours, iris_far_neighbours = find_nearest_iris(x_iris, y_iris, _input, k_iris)
st.title('Predykcja')
st.subheader('Przewidziana klasa: {}'.format(iris_predicted_label))
iris_nearest_neighbours = [[neighbour[1], x_iris[neighbour[1], 0], x_iris[neighbour[1], 1], neighbour[0], y_iris[neighbour[1]]]
                      for neighbour in iris_nearest_neighbours]
iris_nearest_neighbours = pd.DataFrame(iris_nearest_neighbours, columns=['Index', 'PCA1', 'PCA2', 'Distance', 'Klasa'])
st.subheader("Najbliżsi sąsiedzi")
st.dataframe(iris_nearest_neighbours)

# Rysuwanie strzał
iris_far_neighbours = [[neighbour[1], x_iris[neighbour[1], 0], x_iris[neighbour[1], 1], neighbour[0], y_iris[neighbour[1]]]
                  for neighbour in iris_far_neighbours]
iris_far_neighbours = pd.DataFrame(iris_far_neighbours, columns=['Index', 'PCA1', 'PCA2', 'Distance', 'Klasa'])
fig2 = px.scatter(iris_far_neighbours, x='PCA1', y='PCA2', symbol='Klasa', symbol_map={
    0: 'circle',
    1: 'square',
    2: 'triangle-up'})

for index, neighbour in iris_nearest_neighbours.iterrows():
    fig2.add_trace(
        go.Scatter(x=[_input[0], neighbour['PCA1']], y=[_input[1], neighbour['PCA2']], mode='lines+markers',
                   name='{}'.format(neighbour['Index']), showlegend=False)
    )
