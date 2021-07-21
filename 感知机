import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\hp\Desktop\train_binary.csv")
y = df.iloc[0:20000, 0].values
y = np.where(y == 0, -1, 1)
X = df.iloc[0:20000, 1:].values
def perceptron(X,y,iter):
	data = np.mat(X)
	label = np.mat(y).T
	m, n = np.shape(data)
	w = np.zeros(n)
	b = 0
	h = 0.00001
	for k in range(iter):
		for i in range(m):
			xi=data[i]
			yi=label[i]
			if yi*(w*xi.T+b)<=0:
				w=w+h*yi*xi
				b=b+h*yi
	return w,b

w,b=perceptron(X,y,50)

y_test = df.iloc[20000:42000, 0].values
y_test = np.where(y_test == 0, -1, 1)
X_test = df.iloc[20000:42000, 1:].values
data_test=np.mat(X_test)
label_test=np.mat(y_test).T
count=0
for i in range(len(y_test)):
	xi=data_test[i]
	yi=label_test[i]
	if yi*(w*xi.T+b)<=0:
		count+=1
