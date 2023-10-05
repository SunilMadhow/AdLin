from flh import FLH
from sklearn.linear_model import LinearRegression
import numpy as np
from util import *

class OfflinePred(FLH):

	def __init__(self, X, Y, T):
		super().__init__(X, Y, T)
		self.regs = []

	#TODO switch X, Y to instance variables within FLH

	def train_t(self, Xt, Yt, t): # weighted linear least squares on points j = 1, ... t - 1
		W = self.W
		
		#compute weights
		wt = W[t - 1]
		z = []
		for j in range(t):
		    zj = np.sum(wt[:j + 1]*np.array([1/abs(t - s) for s in range(0, j + 1)]))
		    z.append(zj)
    
		weights_t = np.array(z)

		model_t = LinearRegression()
		model_t.fit(Xt.reshape(-1, 1),Yt, weights_t)

		return model_t

	def train(self, T = None):
		if T == None:
			T = self.T

		for t in range(1, T):
			model_t = self.train_t(self.X[:t], self.Y[:t], t)
			self.regs.append(model_t)


	def predict(self, x): # for x_{t-1} < x < x_t evaluates model t-1
		i = np.searchsorted(self.X, x)
		print("i = ", i)
		pred = eval_model(self.regs[i - 1], x)
		return pred


	
