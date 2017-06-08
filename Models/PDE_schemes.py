from auxiliary_functions import solve3Diagonal
import numpy as np

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

# Natural variables Black Scholes Equation vanilla option
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
			pNew = np.array([ (v<g)*self.rho for v,g in zip(stateNew, actualPayoff) ])
			pNew = (stateNew<actualPayoff)*self.rho

			condition = ( np.max([abs(x-y)/max(1,abs(y)) for x, y in zip(stateOld[1:-1], stateNew[1:-1])]) < self.tol) or ( np.array_equal(pNew,pOld) )
			steps += 1
			# print(steps, np.max([abs(x-y)/max(1,y) for x, y in zip(stateOld[1:-1], stateNew[1:-1])]))

		# state update
		self.State = stateNew.copy()
		self.P = pNew.copy()