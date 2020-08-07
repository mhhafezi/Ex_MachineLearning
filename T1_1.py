import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression(fit_intercept=True, normalize=True)


#k=1
for count, i in enumerate([5, 10, 25, 100]):
    titlelist = ['k=1 , i=5', 'k=1 , i=10', 'k=1 , i=25', 'k=1 , i=100']
    x = np.random.rand(i, 1) * 20
    y = 2.358*x - 3.121
    y = y+np.random.normal(scale=3, size=y.shape)
    dataset = np.hstack((x, y))

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    lr_model.fit(x, y)
    print("For {} \nMy Model's Coef is::  ".format(titlelist[count]), lr_model.coef_, "\nMy Model's intercept is::  ",
          lr_model.intercept_)
    score = lr_model.score(x, y)
    print("My Model's score is::  ", score)
    h = lr_model.predict(x)
    MSE = np.mean((y - h) ** 2) / 2
    print("MSE = ", 1 - (2 * MSE / y.var()))
    x_line = np.arange(0, 20, 0.1).reshape(-1, 1)
    y_line = lr_model.predict(x_line)
    plt.plot(x[:, 0], y, 'bo')
    plt.plot(x_line[:, 0], y_line, 'r--')
    plt.title(titlelist[count])
    plt.show()
#k=4
for count, i in enumerate([5, 10, 25, 100]):
    titlelist = ['k=4 , i=5', 'k=4 , i=10', 'k=4 , i=25', 'k=4 , i=100']
    x = np.random.rand(i, 1) * 20
    y = 2.358 * x - 3.121
    y = y + np.random.normal(scale=3, size=y.shape)
    dataset = np.hstack((x, y))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    x_4 = np.hstack((x, x**2, x**3, x**4))
    lr_model.fit(x_4, y)

    print("For {} \nMy Model's Coef is::  ".format(titlelist[count]), lr_model.coef_, "\nMy Model's intercept is::  ",
          lr_model.intercept_)
    score = lr_model.score(x_4, y)
    print("My Model's score is::  ", score)
    h = lr_model.predict(x_4)
    MSE = np.mean((y-h)**2) / 2
    print("MSE = ", 1-(2*MSE / y.var()))
    x_line = np.arange(0, 20, 0.1).reshape(-1, 1)
    x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4))
    y_line = lr_model.predict(x_line)
    plt.plot(x_4[:, 0], y, 'bo')
    plt.plot(x_line[:, 0], y_line, 'r--')
    plt.title(titlelist[count])
    plt.show()

#k=16
for count, i in enumerate([5, 10, 25, 100]):
    titlelist = ['k=16 , i=5', 'k=16 , i=10', 'k=16 , i=25', 'k=16 , i=100']
    x = np.random.rand(i, 1) * 20
    y = 2.358 * x - 3.121
    y = y + np.random.normal(scale=3, size=y.shape)
    dataset = np.hstack((x, y))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    x_16 = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14, x**15,
                      x**16))
    lr_model.fit(x_16, y)
    print("For {} \nMy Model's Coef is::  ".format(titlelist[count]), lr_model.coef_, "\nMy Model's intercept is::  ",
          lr_model.intercept_)
    score = lr_model.score(x_16, y)
    print("My Model's score is::  ", score)
    h = lr_model.predict(x_16)
    MSE = np.mean((y-h)**2) / 2
    print("MSE = ", 1-(2*MSE / y.var()))
    x16_line = np.arange(0, 20, 0.1).reshape(-1, 1)
    x16_line = np.hstack((x16_line, x16_line**2, x16_line**3, x16_line**4, x16_line**5, x16_line**6, x16_line**7,
                          x16_line**8, x16_line**9, x16_line**10, x16_line**11, x16_line**12, x16_line**13, x16_line**14
                          , x16_line**15, x16_line**16))
    y_line = lr_model.predict(x16_line)
    plt.plot(x_16[:, 0], y, 'bo')
    plt.plot(x16_line[:, 0], y_line, 'r--')
    plt.title(titlelist[count])
    plt.show()

