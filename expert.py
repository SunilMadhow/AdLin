from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression

class Expert(ABC):
	#Any expert always expects to receive feedback via update(...) after making a prediction via predict(...)
	@abstractmethod
	def predict(self, x, t): 
		pass

	def update(self, response, t):
		pass

class LinearExpert1d(Expert): # when loss function is specified by a response variable, perform linear regression

	def __init__(self, initial_time):
		self.start = initial_time
		self.predictions = []
		self.history_x =  []
		self.history_y = []

	def __str__(self):
		return "started at time " + str(self.start)

	def eval_model(self, model, point):
		return model.predict(np.array(point).reshape(-1, 1))

	def predict(self, x, t):
		# print("t = ", t)
		# print("x", np.array(self.history_x).reshape(-1, 1))
		# print("pred, ", np.array(self.predictions).reshape(-1, 1))
		# print("y", self.history_y)
		# print("predicting from expert which " + str(self) + "at time " + str(t))


		if np.size(self.history_x) == 0:
			pred =  0
		elif np.size(self.history_x) == 1:
			pred = self.history_y[t-1]	
		else:
			reg = LinearRegression().fit(np.array(self.history_x).reshape(-1, 1), np.array(self.history_y))
			pred = self.eval_model(reg, x)[0]

		self.history_x.append(x)
		self.predictions.append(pred)
		return pred

	def update(self, response, t):
		self.history_y.append(response)



