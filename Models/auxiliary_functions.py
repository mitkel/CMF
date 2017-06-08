import numpy as np
from scipy.stats import norm

def conf_int(arrA, conf_lvl):
	av = np.average(arrA)
	st_dev = np.std(arrA)
	h = 1 - norm.cdf((conf_lvl + 1) / 2)
	conf_int = [av - h * st_dev, av + h * st_dev]
	return {'Price': av, 'ConfInt': conf_int}

def zero_ceil(arr):
	return np.maximum(arr, [0]*len(arr))

def solve3Diagonal(a,b,c,d):
# solves Ax = d for A beeing 3-diagonal matrix
# a - diagonal of length n; b,c - sub and super diagonal of lengths n-1
	n = len(a)
	for i in range(n-1):
		a[i+1] -= c[i]*b[i]/a[i]
		d[i+1] -= d[i]*b[i]/a[i]
	sol = [0]*n
	sol[-1] = d[-1]/a[-1]
	for i in range(n-2,-1,-1):
		sol[i] = (d[i]-c[i]*sol[i+1])/a[i]
	return sol

def powerOption_price(S, K, C, i, r, v, T, type = 'Call'):
	d1 = (np.log(S/(K**(1/i))) + (r + (i-.5)*v**2)*T )/(v*np.sqrt(T))
	d2 = d1 - i*v*np.sqrt(T)
	if type == 'Call':
		return S**i*np.exp( (i-1)*(r+i*v**2/2)*T ) * norm.cdf(d1) - \
			   K*np.exp(-r*T)*norm.cdf(d2)
	else:
		return -S**i*np.exp( (i-1)*(r+i*v**2/2)*T ) * norm.cdf(-d1) + \
			   K*np.exp(-r*T)*norm.cdf(-d2)

def vanilla_payoff(S, K, type = 'Call'):
	phi = {'Call':1, 'Put':-1}[type]
	return np.maximum(phi*(S-K), [0]*len(np.atleast_1d(S)))