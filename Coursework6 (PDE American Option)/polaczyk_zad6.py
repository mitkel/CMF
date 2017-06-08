import matplotlib.pyplot as plt
import numpy as np
from CW6_data import params

# functions
#
#
def vanilla_payoff(S, K, type = 'Call'):
	phi = {'Call':1, 'Put':-1}[type]
	return np.maximum(phi*(S-K), [0]*len(np.atleast_1d(S)))

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

def approximate(x, arr, val):
	i = 0
	while(arr[i]<x):
		i +=1
	j = i
	if abs(arr[i-1]-x) < abs(arr[i]-x):
		i -= 1
	else:
		j -= 1

	return val[i]+(val[i]-val[j])/(arr[i]-arr[j])*(x-arr[i])

# classes
#
#
# PDE scheme with Crank-Nicolson, Euler Implicit & Explicit methods
class PDE_scheme():
	def __init__(self, xBound, x, StartingState, tSteps, endingTime, type ='CN', startingTime = 0):
		self.xBound = xBound
		self.x = np.array(x)
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

# Inversed time American barrier option
class Inversed_American_BSEq(PDE_scheme):
	def __init__(self, xBound, x, StartingState, tSteps, startingTime, endingTime, r, sigma, tol, payoffInv, type = 'CN'):
		super(Inversed_American_BSEq, self).__init__(xBound, x, StartingState, tSteps, endingTime, type, startingTime)
		self.r = r
		self.sigma  = sigma
		self.sigma2 = sigma**2
		self.tol = tol
		self.rho = 1/tol
		self.payoff = payoffInv
		self.P = np.array([0]*self.xLen)

	def goFwd(self):
		# initializing state
		self.t = self.t + self.tIncr
		stateNew = self.State.copy()
		pNew = self.P.copy()

		# step vectors
		actualPayoff = np.array(self.payoff(self.x, self.t))

		gamma = .5 * self.sigma2 * (self.x)**2 / self.xIncr**2
		beta = [np.array([]), np.array([])]
		# j = i - 1
		beta[0] = np.array(self.sigma2*self.x - self.r*self.xIncr > 0) *(-1)
		# j = i + 1
		beta[1] = np.array([1]*self.xLen)
		beta *= .5* self.r/self.xIncr * self.x

		steps = 0
		condition = False
		while not condition:
			stateOld = stateNew.copy()
			pOld = pNew.copy()

			# calculating right hand side of the equation
			RHS = stateOld[1:-1].copy()
			RHS -= (1-self.theta)*self.tIncr*(
					self.r*stateOld[1:-1] -
					# j = i - 1
					np.array([(c+b)*(y-x) for c,b,x,y in zip(gamma[1:-1], beta[0][1:-1], stateOld[1:-1], stateOld[:-2])]) -
					# j = i + 1
					np.array([(c+b)*(y-x) for c,b,x,y in zip(gamma[1:-1], beta[1][1:-1], stateOld[1:-1], stateOld[2:])])
				)
			RHS += + pOld[1:-1] * actualPayoff[1:-1]

			# + boundary conditions
			RHS[0 ] += self.theta*self.tIncr*(gamma[1] + beta[0][1])*self.xBound(self.x[0], self.t)
			RHS[-1] += self.theta*self.tIncr*(gamma[-2] + beta[1][-2])*self.xBound(self.x[-1], self.t)

			# calculating 3-diagonal matrix from the left hand side of the equation
			LHS = [[] for _ in range(3)] #lower-diagonal, diagonal, upper-diagonal
			LHS[0] = -self.theta*self.tIncr * (gamma[2:-1] + beta[0][2:-1])
			LHS[2] = -self.theta*self.tIncr * (gamma[1:-2] + beta[1][1:-2])
			LHS[1] = 1. + pOld[1:-1] + self.theta*self.tIncr * self.r
			LHS[1] += self.theta*self.tIncr * (2*gamma[1:-1] + beta[0][1:-1] + beta[1][1:-1])

			# solving equation and updating state
			stateNew[1:-1] = solve3Diagonal(LHS[1], LHS[0], LHS[2], RHS)
			pNew = (stateNew < actualPayoff)*self.rho

			condition = ( np.max([abs(x-y)/max(1,abs(y)) for x, y in zip(stateOld[1:-1], stateNew[1:-1])]) < self.tol) or ( np.array_equal(pNew,pOld) )
			steps += 1
			# print(steps, np.max([abs(x-y)/max(1,y) for x, y in zip(stateOld[1:-1], stateNew[1:-1])]))

		# state update
		self.State = stateNew.copy()
		self.P = pNew.copy()

# main function
def American_DO(S0, r, sigma, K, T, Mt, Mx, xmin, xmax, tol, X):

	try:
		resp = input('What do you want?:\n 1.Crank-Nicolson Call Option\n 2.Implicit Call Option\n 3.Crank-Nicolson Put Option\n 4.Implicit Put Option\n 5.Get me out of here\n')
		if int(resp) == 5:
			return 0
		scheme_type = {0: 'Implicit', 1: 'CN'}[int(resp)%2]
		option_type = (resp in ['1','2'])*'Call' + (resp in ['3','4'])*'Put'
	except (KeyError, ValueError, NameError):
		print('Incorrect input.')
		return 0

	# setting boundary conditions
	def xBound(s, t):
		return vanilla_payoff(s,K, option_type)

	# setting starting parameters
	xmin = max(X,xmin)
	S = np.linspace(xmin, xmax + (xmax - xmin) / Mx, Mx)
	state0 = xBound(S,0)

	# initializing model
	model = Inversed_American_BSEq(xBound, S, state0, Mt, 0, T, r, sigma, tol, xBound, type=scheme_type)

	# running scheme
	res = [state0]
	for Y in model.run():
		res = np.append(res.copy(), [Y], axis=0)
	V = res[-1]

	# plotting results
	plt.clf()
	plt.gca().set_xlim(xmin, xmax)
	plt.plot(S, V)
	plt.savefig(option_type + '_' + scheme_type + '.png')
	plt.clf()

	# returning answer
	ans = approximate(S0, S, V)
	return { 'V':ans }

print(American_DO(**params))
