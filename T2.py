import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
MSE_train = []
MSE_test = []
score_train = []
score_test = []
lr_model = LinearRegression(fit_intercept=True, normalize=True)
x = np.random.rand(200, 1) * 20
y = 2*(x**4) - x**3 + 3*(x**2) - 5*x + 4
e = np.random.normal(scale=3, size=y.shape)
y = y + e
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

split = int(x.shape[0] * 0.8)
x_train = x[:split]
y_train = y[:split]
x_test = x[split:]
y_test = y[split:]

lr_model.fit(x_train, y_train)

h_train1 = lr_model.predict(x_train)
h_test1 = lr_model.predict(x_test)

MSE_train.append(np.mean((y_train-h_train1)**2) / 2)
MSE_test.append(np.mean((y_test-h_test1)**2) / 2)

print('For degree 1:')
print("TRAIN's MSE = \t", lr_model.score(x_train, y_train))
print("TEST's MSE = \t", lr_model.score(x_test, y_test))

score_train.append(lr_model.score(x_train, y_train))
score_test.append(lr_model.score(x_test, y_test))

# no overfit nor under fit
x_line = np.arange(0, 20, 0.1).reshape(-1, 1)
y_line = lr_model.predict(x_line)

plt.plot(x_line[:, 0], y_line, 'r--')
plt.plot(x_train,  y_train, 'bo')
plt.plot(x_test,  y_test, 'co')
plt.show()

#degree=2
x = np.random.rand(200, 1) * 20
y = 2*x**4 - x**3 + 3*x**2 - 5*x + 4
x = np.hstack((x, x**2))
e = np.random.normal(scale=3, size=y.shape)
y = y + e
y.reshape(-1,1)

split = int(x.shape[0] * 0.8)
x_train = x[:split]
y_train = y[:split]
x_test = x[split:]
y_test = y[split:]

lr_model.fit(x_train,y_train)
h_train2 = lr_model.predict(x_train)
h_test2 = lr_model.predict(x_test)
print('For degree 2:')
print("TRAIN's MSE = \t", lr_model.score(x_train, y_train))
print("TEST's MSE = \t", lr_model.score(x_test, y_test))
score_train.append(lr_model.score(x_train, y_train))
score_test.append(lr_model.score(x_test, y_test))
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2))
y_line = lr_model.predict(x_line)
MSE_train.append(np.mean((y_train-h_train2)**2) / 2)
MSE_test.append(np.mean((y_test-h_test2)**2) / 2)
plt.plot(x_line[:,0], y_line, 'r--')
plt.plot(x_train[:,0],  y_train, 'bo')
plt.plot(x_test[:,0],  y_test, 'co')
plt.show()
#degree=4
x = np.random.rand(200, 1) * 20
y = 2*x**4 - x**3 + 3*x**2 - 5*x + 4
x = np.hstack((x, x**2, x**3, x**4))
e = np.random.normal(scale=3, size=y.shape)
y = y + e
y.reshape(-1,1)

split = int(x.shape[0] * 0.8)
x_train = x[:split]
y_train = y[:split]
x_test = x[split:]
y_test = y[split:]
lr_model.fit(x_train,y_train)
score_train.append(lr_model.score(x_train, y_train))
score_test.append(lr_model.score(x_test, y_test))

h_train3 = lr_model.predict(x_train)
h_test3 = lr_model.predict(x_test)
print('For degree 4:')
print("TRAIN's MSE = \t", lr_model.score(x_train, y_train))
print("TEST's MSE = \t", lr_model.score(x_test, y_test))
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4))
y_line = lr_model.predict(x_line)
MSE_train.append(np.mean((y_train-h_train3)**2) / 2)
MSE_test.append(np.mean((y_test-h_test3)**2) / 2)
plt.plot(x_line[:,0], y_line, 'r--')
plt.plot(x_train[:,0],  y_train, 'bo')
plt.plot(x_test[:,0],  y_test, 'co')
plt.show()

#degree=8
x = np.random.rand(200, 1) * 20
y = 2*x**4 - x**3 + 3*x**2 - 5*x + 4
x = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8))
e = np.random.normal(scale=3, size=y.shape)
y = y + e
y.reshape(-1,1)

split = int(x.shape[0] * 0.8)
x_train = x[:split]
y_train = y[:split]
x_test = x[split:]
y_test = y[split:]

lr_model.fit(x_train,y_train)
h_train = lr_model.predict(x_train)
h_test = lr_model.predict(x_test)
MSE_train.append(np.mean((y_train-h_train)**2) / 2)
MSE_test.append(np.mean((y_test-h_test)**2) / 2)
print('For degree 8:')
print("TRAIN's MSE = \t", lr_model.score(x_train, y_train))
print("TEST's MSE = \t", lr_model.score(x_test, y_test))
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4, x_line**5, x_line**6, x_line**7, x_line**8))
y_line = lr_model.predict(x_line)
score_train.append(lr_model.score(x_train, y_train))
score_test.append(lr_model.score(x_test, y_test))

plt.plot(x_line[:,0], y_line, 'r--')
plt.plot(x_train[:,0],  y_train, 'bo')
plt.plot(x_test[:,0],  y_test, 'co')
plt.show()


#degree=16
x = np.random.rand(200, 1) * 20
y = 2*x**4 - x**3 + 3*x**2 - 5*x + 4
x = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14, x**15, x**16))
e = np.random.normal(scale=3, size=y.shape)
y = y + e
y.reshape(-1,1)

split = int(x.shape[0] * 0.8)
x_train = x[:split]
y_train = y[:split]
x_test = x[split:]
y_test = y[split:]

lr_model.fit(x_train,y_train)

print('For degree 16:')
print("TRAIN's MSE = \t", lr_model.score(x_train, y_train))
print("TEST's MSE = \t", lr_model.score(x_test, y_test))
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4, x_line**5, x_line**6, x_line**7, x_line**8, x_line**9,
                    x_line**10, x_line**11, x_line**12, x_line**13, x_line**14, x_line**15, x_line**16))
y_line = lr_model.predict(x_line)
h_train4 = lr_model.predict(x_train)
h_test4 = lr_model.predict(x_test)
MSE_train.append(np.mean((y_train-h_train4)**2) / 2)
MSE_test.append(np.mean((y_test-h_test4)**2) / 2)
score_train.append(lr_model.score(x_train, y_train))
score_test.append(lr_model.score(x_test, y_test))
plt.plot(x_line[:,0], y_line, 'r--')
plt.plot(x_train[:,0],  y_train, 'bo')
plt.plot(x_test[:,0],  y_test, 'co')
plt.show()

deg = [1, 2, 4, 8, 16]
plt.plot(deg, MSE_train, 'go-', label='MSE_train')
plt.plot(deg, MSE_test, 'ro-', label='MSE_TEST')
plt.legend()
plt.show()



plt.plot(deg, score_train, 'go-', label='score_train')
plt.plot(deg, score_test, 'ro-', label='score_TEST')
plt.legend()
plt.show()