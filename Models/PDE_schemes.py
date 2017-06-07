from auxiliary_functions import solve3Diagonal
import numpy as np

# PDE scheme with Crank-Nicolson, Euler Implicit & Explicit methods
class PDE_scheme():
	def __init__(self, xBound, x, StartingState, tSteps, endingTime, type = 'E', startingTime = 0):
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
		self.t = self.t + self.tIncr
		stateOld = self.State
		stateNew = self.State
		stateNew[0] = self.xBound(self.x[0], self.t)
		stateNew[-1] = self.xBound(self.x[-1], self.t)
		pOld = self.P
		pNew = self.P
		actualPayoff = np.array(self.payoff(self.x, self.t))

		gamma = .5 * self.sigma2 * (self.x)**2 / self.xIncr**2
		beta = [np.array([]), np.array([])]
		# j = i - 1
		beta[0] = np.array([(self.sigma2*x-self.r*self.xIncr>0)*(-1) for x in self.x])
		# j = i + 1
		beta[1] = np.array([(self.sigma2*x+self.r*self.xIncr>0)*1 +
							(self.sigma2*x+self.r*self.xIncr<=0)*2
							for x in self.x])
		beta = .5* self.r/self.xIncr * self.x * beta

		condition = True
		while condition:

			RHS = stateOld[1:-1] + \
				pOld[1:-1] * actualPayoff[1:-1] - \
				self.theta*self.tIncr*(
					r*stateOld[1:-1] -
					# j = i - 1
					np.array([(c+b)*(y-x) for c,b,x,y in zip(gamma[1:-1], beta[0][1:-1], stateOld[1:-1], stateOld[:-2])]) -
					# j = i + 1
					np.array([(c+b)*(y-x) for c,b,x,y in zip(gamma[1:-1], beta[1][1:-1], stateOld[1:-1], stateOld[2:])])
				)
			LHS = np.array(np.array([]),np.array([]),np.array([])) #lower-diagonal, diagonal, upper-diagonal
			LHS[0] = - gamma[2:-1] - beta[0][2:-1]
			LHS[2] = - gamma[1:-2] - beta[1][1:-2]
			LHS[1] = 1 + pOld[1:-1]
			LHS[1][:-1] -= LHS[2]
			LHS[1][1:] -= LHS[0]

			stateNew[1:-1] = solve3Diagonal(LHS[1], LHS[0], LHS[2], RHS)
			pOld = pNew
			pNew = np.array([ (g<v)*self.rho for v,g in zip(stateNew, actualPayoff) ])

			condition = ( np.max([abs(x-y)/max(1,y) for x,y in zip(stateOld,stateNew)]) < tol ) \
						or ( pNew == pOld )

		# state update
		self.State = stateNew
		self.P = pNew