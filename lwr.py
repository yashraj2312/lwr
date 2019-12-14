from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from math import pi, ceil
import numpy as np

np.random.seed(2)


def lowess(x, y, f=2. / 3., iter=3):
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]

    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3

    yest = np.zeros(n)
    delta = np.ones(n)
    for _ in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = np.linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest


n = 100
x = np.linspace(0, 2 * pi, n)
y = np.sin(x) + 0.3 * np.random.randn(n)

f = 0.25
yest = lowess(x, y, f=f, iter=3)

'''-----------------------------------------'''
n = 100
x = np.linspace(0, 2 * pi, n)
y = np.sin(x) + 0.2 * np.random.randn(n)
yActual = y[:]

for _ in range(15):
    neigh = KNeighborsRegressor(n_neighbors=5)
    neigh.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    newY = neigh.predict(x.reshape(-1, 1))
    y = newY

plt.plot(x, yActual, label='Noisy')
plt.plot(x, yest, label='Lowess')
plt.plot(x, newY, label='KNN Scikit')
plt.legend(loc='upper right')
plt.show()