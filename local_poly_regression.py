import numpy as np

def boxcar(x, Xi, bandwidth):
	return abs(x - Xi)/bandwidth <= 1/2

def data_matrix(X):
	return np.concatenate((np.ones((X.size, 1)), X.reshape(X.size, 1)), axis = 1)

def local_linear(X, Y, bandwidth, x):
	B = data_matrix(X)
	Omega = np.diag([boxcar(x, Xi, bandwidth) for Xi in X])
	b = np.array([1, x])
	return b.T@np.linalg.inv(B.T@Omega@B)@B.T@Omega@Y