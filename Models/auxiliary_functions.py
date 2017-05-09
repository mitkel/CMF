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