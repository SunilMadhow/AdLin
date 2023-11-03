import numpy as np
from math import factorial

def div_diff(f, z):
	assert f.size == z.size
	print("f = ", f)
	if f.size == 1:
		return f[0]
	return (div_diff(f[1:], z[1:]) - div_diff(f[:-1], z[:-1]))/(z[-1]- z[0])

def disc_der(fz, z,fx, x, k):
	i = np.searchsorted(z, x)
	if i == 0:
		return fx
	elif i < k:
		return factorial(i)*div_diff(np.append(fz[:i], fx), np.append(z[:i], x))
	elif i >= k:
		return factorial(k)*div_diff(np.append(fz[i - k: i], fx), np.append(z[i-k:i], x))

def h(k, j, z, x): #evaluates h^k_j at x where {h^k} is the k-th degree falling factorial basis
	# j - 2 is how many points we use
	if j < k + 2:
		return np.prod(np.ones(j-2)*x - z[:j-2])/factorial(j-1)
	else:
		indicator = (x > z[j - 2])
		return indicator*np.prod(np.ones(k)*x - z[j - k - 1:j - 2])/factorial(k)

def interpolation(fz, z, k): #returns list, a, of length n with a[j] = coefficient of h^k_j
	a = np.array(f.size)
	for j in range(0, k + 1):
		a[j] = disc_der(f, z, fz[j], z[j])
	for j in range(k+1, z.size):
		a[j] = disc_der(f, z, fz[j], z[j])*(z[j] - z[j - k - 1])
	return a

def interpolate(fz, z,  x, k, a = None): #eval interpolation @x
	H = np.array([h(k, j, z, x) for j in range(0, n)])
	if a == None:
		a = interpolation(fz, z, k)
	return np.inner(a, H)









