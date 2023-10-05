from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression

class Expert(ABC):
	@abstractmethod
	def predict(self, x, t):
		pass

	def update(self, response, t):
		pass

class LinearExpert1d(Expert): # when loss function is specified by a response variable, perform linear regression

	def __init__(self, initial_time):
		self.start = initial_time
		self.predictions = np.array([])
		self.history_x =  np.array([])
		self.history_y = np.array([])


	def predict(self, x, t):
		if np.size(self.history_y) == 0:
			np.append(self.predictions, 0)
			return 0
		elif np.size(self.history_y) == 1:
			np.append(self.predictions, self.history_y[t - 1])
			return self.history_y[t-1]
		else:
			reg = LinearRegression().fit(self.history_x, self.history_y)
			pred = reg.predict(x)
			np.append(self.predictions, pred)
			return pred


		np.append(self.history_x, x)

	def update(self, response, t):
		np.append(self.history_y, response)



