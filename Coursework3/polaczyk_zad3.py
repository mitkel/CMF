from CW3_data import params1, params2
import numpy as np
import numpy.random as npr
from scipy.stats import norm

class MC_European_model():
	def __init__(self, S0, T, StepsNo, r, SimNo):
		self.stepsNo = StepsNo
		self.maturity = T
		self.stepSize = T / StepsNo
		self.r = r
		self.increment = np.sqrt(self.stepSize)
		self.SimNo = SimNo
		self.state = [S0]*SimNo

	def goFwd(self):
		raise NotImplementedError

	def Finish(self):
		raise NotImplementedError

	def run(self):
		for _ in range(self.stepsNo):
			self.goFwd()
		return self.Finish()

class Heston_MC_model(MC_European_model):
	def __init__(self, S0, V0, T, K, r, a, b, sigma, rho, M, n, CallPut):
		super(Heston_MC_model, self).__init__(S0, T, n, r, M)
		self.state = {'S':np.array([S0]*self.SimNo),
					  'V':np.array([V0]*self.SimNo),
					  'Z':np.log(np.array([S0]*self.SimNo)),
					  't':0}
		self.a = a
		self.b = b
		self.sigma = sigma
		self.corr = rho
		self.strike = K
		self.type = CallPut

	def Finish(self, conf_lvl = .95):
		sol = {}
		for type in ['S','Z']:
			if type == 'Z':
				tmp = np.exp(self.state[type])
			else:
				tmp = self.state[type]

			disc_payoffs = np.exp(-self.r*self.state['t'])*self.payment(tmp)
			price = np.average(disc_payoffs)
			st_dev = np.sqrt(np.var(disc_payoffs)/self.SimNo)
			h = 1 - norm.ppf((conf_lvl + 1) / 2)
			conf_int = [price - h*st_dev, price + h*st_dev]
			sol[type] = {'Price': price, 'ConfInt':conf_int}

		return sol

	def goFwd(self):
		randIncr = npr.multivariate_normal([0,0], [[self.stepSize, self.corr*self.stepSize],
												   [self.corr*self.stepSize, self.stepSize]],
										   size=self.SimNo)

		# self.StockStep(np.array([x[0] for x in randIncr])) #krok pominiety, bo logS bardziej wydajne
		self.logStep(np.array([x[0] for x in randIncr]))
		self.VolStep(np.array([x[1] for x in randIncr]))
		self.state['t'] += self.stepSize

	def StockStep(self, randIncr):
		self.state['S'] += self.state['S']*(self.r*self.stepSize +
											np.sqrt(self.zero_ceil(self.state['V']))*randIncr)

	def logStep(self, randIncr):
		self.state['Z'] += (self.r-.5*self.state['V'])*self.stepSize + \
						   np.sqrt(self.zero_ceil(self.state['V']))*randIncr

	def VolStep(self, randIncr):
		self.state['V'] += self.a*(self.b-self.state['V'])*self.stepSize +\
						   self.sigma*np.sqrt(self.zero_ceil(self.state['V']))*randIncr +\
						   .25*self.sigma**2*(randIncr**2-self.stepSize)

	def payment(self, S):
		phi = ((self.type == 'Put') - .5) * 2
		return self.zero_ceil( phi * (self.strike - S) )

	def zero_ceil(self, arr):
		return np.maximum(arr, [0]*len(arr))

def Heston_call_MC(**kwargs):
	model = Heston_MC_model(**kwargs, CallPut='Call')
	res = model.run()
	return res['Z']

print(Heston_call_MC(**params1))
print(Heston_call_MC(**params2))