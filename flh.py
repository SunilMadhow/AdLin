from expert import LinearExpert1d
import numpy as np

class FLH:
	lr = .5

	def __init__(self, T, expert = LinearExpert1d(1)):
		self.T = T
		self.t = 1
		self.experts = [expert]
		self.weights = [1]
		self.loss = 0
		self.predictions = []

	def __step(self, x, y):
		t = self.t

		x_experts = []
		for e in self.experts:
			# print("e.history_y = ", e.history_y)
			x_experts.append(e.predict(x, t - e.start))
			e.update(y, t)

		pred = np.dot(np.array(self.weights), np.array(x_experts))
		self.predictions.append(pred)
		
		loss_t = (pred - y)**2
		self.loss = self.loss + loss_t

		l_experts = np.square(np.array(x_experts) - y)
		# print("l_experts = ", l_experts)

		new_weights = np.exp(-self.lr*l_experts)*np.array(self.weights)
		new_weights =  ((new_weights / np.sum(new_weights))*(1 - 1/(t + 1))).tolist()
		new_weights.append(1/(t+1))
		self.weights = new_weights


		new_exp = LinearExpert1d(self.t + 1)
		self.experts.append(new_exp)

		self.t = self.t + 1

	def run(self, X, Y):
		W = []
		for j in range(self.T):
			W.append(self.weights + (self.T - self.t)*[0])
			print("W = ", W)
			self.__step(X[j], Y[j])
			
		print("loss = ", self.loss)
		return np.array(W)





