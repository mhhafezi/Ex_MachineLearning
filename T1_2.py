import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression(fit_intercept=True, normalize=True)
#k=10
for count, i in enumerate([5, 10, 25, 100]):
    x = np.random.rand(i, 1) * 20
    x = x.reshape(-1, 1)
    y = 2.358 * x - 3.121
    y = y + np.random.normal(scale=3, size=y.shape)
    y = y.reshape(-1, 1)
    lr_model.fit(x, y)
    h = lr_model.predict(x)
    MSE = np.mean((y - h) ** 2) / 2
    print("MSE = ", 1 - (2 * MSE / y.var()))
    MSE_list = []
    MSE_list.append(MSE)

    for j in range(1, 11):
        x = np.hstack((x, x**(j+1)))
        y = 2.358 * x - 3.121
        y = y + np.random.normal(scale=3, size=y.shape)
        y = y.reshape(-1, 1)
        lr_model.fit(x, y)
        h = lr_model.predict(x)
        MSE = np.mean((y-h)**2) / 2
        print("MSE = ", 1-(2*MSE / y.var()))
        MSE_list.append(MSE)
        plt.plot(range(1,11), MSE_list, 'bo')
        plt.xlabel("Power")
        plt.ylabel('MSE')
        plt.show()
