import matplotlib.pyplot as plt
import numpy as np
from CW5_data import params
from scipy.stats import norm

# auxiliary functions
def payoff(S, K, C, i):
	if type(S) in [float, int, np.float64]:
		return min(max(S ** i - K, 0), C)
	else:
		X = [max(s ** i - K, 0) for s in S]
		return np.array([min(x, C) for x in X])

def solve3Diagonal(a, b, c, d):
	# solves Ax = d for A beeing 3-diagonal matrix
	# a - diagonal of length n; b,c - sub and super diagonal of lengths n-1
	n = len(a)
	for i in range(n - 1):
		a[i + 1] -= c[i] * b[i] / a[i]
		d[i + 1] -= d[i] * b[i] / a[i]
	sol = [0] * n
	sol[-1] = d[-1] / a[-1]
	for i in range(n - 2, -1, -1):
		sol[i] = (d[i] - c[i] * sol[i + 1]) / a[i]
	return sol

def approximate(x, arr, val, der):
	i = 0
	while(arr[i]<x):
		i +=1
	j = i
	if abs(arr[i-1]-x) < abs(arr[i]-x):
		i -= 1
	else:
		j -= 1

	return [val[i]+der[i]*(x-arr[i]),
			(der[i]*abs(x-arr[i]) + der[j]*abs(x-arr[j]))/abs(arr[i]-arr[j])]

def powerOption_price(S, K, C, i, r, sigma, T, type = 'Call'):
	d1 = (np.log(S/(K**(1/i))) + (r + (i-.5)*sigma**2)*T )/(sigma*np.sqrt(T))
	d2 = d1 - i*sigma*np.sqrt(T)
	if type == 'Call':
		return S**i*np.exp( (i-1)*(r+i*sigma**2/2)*T ) * norm.cdf(d1) - \
			   K*np.exp(-r*T)*norm.cdf(d2)
	else:
		return -S**i*np.exp( (i-1)*(r+i*sigma**2/2)*T ) * norm.cdf(-d1) + \
			   K*np.exp(-r*T)*norm.cdf(-d2)

# classes
# PDE scheme with Crank-Nicolson, Euler Implicit & Explicit methods
class PDE_scheme():
	def __init__(self, xBound, x, StartingState, tSteps, endingTime, type = 'E', startingTime = 0):
		self.xBound = xBound
		self.x = x
		self.State = np.array(StartingState)
		self.xLen = len(x)
		self.xIncr = (x[-1] - x[1])/(self.xLen-1) #x[0]=xmin is sometimes -infinity

		self.tSteps = tSteps
		self.T = endingTime
		self.t = startingTime
		self.tIncr = (endingTime - startingTime)/tSteps

		if type in ['Explicit', 'CN', 'Implicit']:
			self.type = type
		else:
			print("Unknown method: choose from Implicit, Explicit and CN. Default CN chosen.")
			self.type = 'CN'

		theta = {
			'Explicit':0.,
			'CN':.5,
			'Implicit':1.}[self.type]
		self.theta = theta

	def goFwd(self):
		raise NotImplementedError

	def run(self):
		for _ in range(self.tSteps):
			self.goFwd()
			yield(self.State)
		# return self.Finish()

# Heat Equation y_t = y_xx
class HeatEq(PDE_scheme):
	def __init__(self, xBound, x, StartingState, tSteps, endingTime, type = 'CN', startingTime = 0):
		super(HeatEq, self).__init__(xBound, x, StartingState, tSteps, endingTime, type, startingTime)

	def goFwd(self):
		self.t = self.t + self.tIncr

		lam = self.tIncr/((self.xIncr)**2)

		# preparation of equation matrices
		a = [1+2*self.theta*lam]*(self.xLen-2)
		b = [-self.theta*lam]*(self.xLen-3)
		c = [-self.theta*lam]*(self.xLen-3)

		d = [0]*(self.xLen-2)
		d[0]  = (1-self.theta)*lam*self.State[0]
		d[-1] = (1-self.theta)*lam*self.State[-1]

		self.State[0]  = self.xBound(self.x[0], self.t)
		self.State[-1] = self.xBound(self.x[-1], self.t)
		d[0]  += self.theta*lam*self.State[0]
		d[-1] += self.theta*lam*self.State[-1]

		d += (1-2*(1-self.theta)*lam)*self.State[1:-1]
		d[:-1] += lam*(1-self.theta)*self.State[2:-1]
		d[1:]  += lam*(1-self.theta)*self.State[1:-2]

		# forward-backward Gauss elimination
		self.State[1:-1] = np.array(solve3Diagonal(a,b,c,d))

# Natural variables Black Scholes Equation
class Natural_variables_BSEq(PDE_scheme):
	def __init__(self, xBound, x, StartingState, tSteps, startingTime, endingTime, r, sigma, type = 'CN'):
		super(Natural_variables_BSEq, self).__init__(xBound, x, StartingState, tSteps, endingTime, type, startingTime)
		self.r = r
		self.sigma  = sigma
		self.sigma2 = sigma**2

	def goFwd(self):
		self.t = self.t + self.tIncr

		# preparation of the equation matrices
		I1 = np.array([x    for x in np.arange(1,self.xLen-1)])
		I2 = np.array([x**2 for x in np.arange(1,self.xLen-1)])

		B = [1] * (self.xLen-2) - self.theta*self.tIncr*(self.r + self.sigma2*I2)
		A = .5*self.theta*self.tIncr*(self.sigma2*I2 - self.r*I1)
		C = .5*self.theta*self.tIncr*(self.sigma2*I2 + self.r*I1)

		b = [1] * (self.xLen-2) + (1-self.theta)*self.tIncr*(self.r + self.sigma2*I2)
		a = -.5*(1-self.theta)*self.tIncr*(self.sigma2*I2 - self.r*I1)
		c = -.5*(1-self.theta)*self.tIncr*(self.sigma2*I2 + self.r*I1)

		d = [0] * (self.xLen - 2)
		d[0]  = a[0]  * self.State[0]
		d[-1] = c[-1] * self.State[-1]
		self.State[0] = self.xBound(self.x[0], self.t)
		self.State[-1] = self.xBound(self.x[-1], self.t)
		d[0]  -= A[0] * self.State[0]
		d[-1] -= C[-1] * self.State[-1]

		d += b * self.State[1:-1]
		d[:-1] += c[:-1] * self.State[2:-1]
		d[1:]  += a[1:]  * self.State[1:-2]

		# forward-backward Gauss elimination
		self.State[1:-1] = np.array(solve3Diagonal(B, A[1:], C[:-1], d))

# main function
def CapPowerCallPDE(S0, r, sigma, K, T, C, i, Mt, Mx, xmin, xmax):
	natural = '0'
	while natural == '0':
		try:
			resp = input('Choose variable type:\n 1.Natural\n 2.Transformed\n 3.Close Program\n')
			natural = {'1': True, '2': False, '3': 'exit'}[resp]
		except KeyError:
			print('Incorrect input.')
	if natural == 'exit':
		return 0

	type = '0'
	while type == '0':
		try:
			resp = input('Choose scheme:\n 1.Crank-Nicolson\n 2.Implicit\n 3.Close Program\n')
			type = {'1': 'CN', '2': 'Implicit', '3': 'exit'}[resp]
		except KeyError:
			print('Incorrect input.')
	if type == 'exit':
		return 0

	S = np.linspace(xmin, xmax + (xmax - xmin) / Mx, Mx)
	X = np.array([np.log(0)] + list(np.linspace(np.log(S[1]), np.log(S[-1]) + np.log(S[-1] / S[1]) / Mx, Mx - 1)))
	backTransform = 1

	# setting initial parameters
	if natural:
		def xBound(s, t):
			if s == S[0]:
				return 0
			else:
				return np.exp(-(T - t) * r) * C

		state0 = np.array(payoff(S, K, C, i))
		model = Natural_variables_BSEq(xBound, S, state0, Mt, T, 0, r, sigma, type=type)
	else:
		lam = 2 * r / sigma ** 2
		a = (lam - 1) / 2
		b = .25 * (lam - 1) ** 2 + lam

		def xBound(x, tau):
			if x == X[0]:
				return 0
			else:
				return np.exp(a * x + (b - lam) * tau) * C

		state0 = np.array(np.exp(a * X) * payoff(np.exp(X), K, C, i))
		model = HeatEq(xBound, X, state0, Mt, T * sigma ** 2 / 2, type)
		backTransform = np.exp(-a * X - b * T * sigma ** 2 / 2)

	# running scheme
	res = [state0]
	for Y in model.run():
		res = np.append(res, [Y], axis=0)
	res = backTransform * res
	V = res[-1]
	Delta = [0] * len(S)
	for j in np.arange(1, len(S) - 1):
		Delta[j] = (V[j + 1] - V[j - 1]) / (2 * (S[1] - S[0]))

	# plotting results
	plt.clf()
	plt.gca().set_xlim(xmin, xmax)
	if natural:
		tmp = S
		text = 'natural'
	else:
		tmp = np.exp(X)
		text = 'transformed'

	plt.plot(tmp, V)
	plt.savefig(text + '_' + type + '.png')
	plt.clf()
	plt.plot(tmp, Delta)
	plt.savefig(text + '_Delta_' + type + '.png')
	plt.clf()

	ans = approximate(S0, tmp, V, Delta)
	return {'V':ans[0], 'Delta':ans[1], 'err:':abs(ans[0] - powerOption_price(S0,K,C,i,r,sigma,T)) }

print(CapPowerCallPDE(**params))
