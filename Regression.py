from sklearn.datasets import make_regression


x, y = make_regression(
        n_samples=100,
        n_features=2,
        n_informative=2,
        random_state=3
    )
print(x, y, sep="\n\n")
