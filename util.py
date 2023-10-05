import numpy as np

def eval_model(model, point):
		return model.predict(np.array(point).reshape(-1, 1))[0]