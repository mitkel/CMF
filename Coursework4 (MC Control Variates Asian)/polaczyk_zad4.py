import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import norm
from CW4_data import params

def conf_int(arrA, conf_lvl):
	av = np.average(arrA)
	st_dev = np.std(arrA)
	h = 1 - norm.cdf((conf_lvl + 1) / 2)
	conf_int = [av - h * st_dev, av + h * st_dev]
	return {'Price': av, 'ConfInt': conf_int}

def zero_ceil(arr):
	return np.maximum(arr, [0]*len(arr))

class MC_model():
	def __init__(self, S0, T, StepsNo, r, SimNo, K, CallPut):
		self.stepsNo = StepsNo
		self.maturity = T
		self.stepSize = T / StepsNo
		self.r = r
		self.increment = np.sqrt(self.stepSize)
		self.SimNo = SimNo
		self.state = [S0]*SimNo
		self.strike = K
		self.type = CallPut

	def goFwd(self):
		raise NotImplementedError

	def Finish(self):
		raise NotImplementedError

	def run(self):
		for _ in range(self.stepsNo):
			self.goFwd()
		return self.Finish()

	def payment(self, S):
		phi = ((self.type == 'Put') - .5) * 2
		return zero_ceil( phi * (self.strike - S) )

class asian_MC_model(MC_model):
	def __init__(self, S0, T, K, r, sigma, M, n, CallPut = 'Call', conf_lvl = .95):
		super(asian_MC_model, self).__init__(S0, T, n, r, M, K, CallPut)
		self.strike = K
		self.sigma = sigma
		self.state = {'S':np.array([S0] * self.SimNo),
					  'Z':np.log(np.array([S0] * self.SimNo)),
					  'ArtMean':np.array([S0] * self.SimNo),
					  'GeoMean': np.array([np.log(S0)] * self.SimNo),
					  't':0}
		self.conf_lvl = conf_lvl
		self.truePrice = {}

		sigma2 = sigma*np.sqrt((2*self.stepsNo+1)/(6*(self.stepsNo+1)))
		rho = .5*(self.r - .5*sigma**2 + sigma2**2)
		d1 = ( np.log(S0/K) + (rho + .5*sigma2**2) *T ) / (sigma2 * np.sqrt(T))
		d2 = ( np.log(S0/K) + (rho - .5*sigma2**2) *T ) / (sigma2 * np.sqrt(T))
		self.truePrice['Disc'] = np.exp(-r * T) * (
			S0*np.exp(T*rho)*norm.cdf(d1) - K*norm.cdf(d2))

		rho = .5 * self.r - sigma**2/12
		d1 = ( np.log(S0/K) + (rho + .5*sigma**2/3) * T) / ((sigma/np.sqrt(3)) * np.sqrt(T))
		d2 = ( np.log(S0/K) + (rho - .5*sigma**2/3) * T) / ((sigma/np.sqrt(3)) * np.sqrt(T))
		self.truePrice['Cont'] = np.exp(-r * T) * (
			S0*np.exp(T*rho)*norm.cdf(d1) - K*norm.cdf(d2))


	def goFwd(self):
		randIncr = npr.normal(0, np.sqrt(self.stepSize), size=self.SimNo)
		self.state['S'] += self.state['S']*(self.r*self.stepSize +
											self.sigma*randIncr)
		self.state['Z'] += (self.r-.5*self.sigma**2)*self.stepSize + \
						   self.sigma*randIncr

		self.state['t'] += 1

		self.state['ArtMean'] = (self.state['ArtMean']*self.state['t']+
								 self.state['S'])/\
								(self.state['t']+1)
		self.state['GeoMean'] = (self.state['GeoMean']*self.state['t']+
								(self.state['Z']))/\
								(self.state['t']+1)

	def Finish(self):
		self.state['GeoMean'] = np.exp(self.state['GeoMean'])
		disc_payoffs = {}
		for type in ['ArtMean', 'GeoMean']:
			disc_payoffs[type] = np.exp(-self.r*self.maturity)*self.payment(self.state[type])

		sol = {}
		for type in ['Disc', 'Cont']:
			corr = np.arange(-2,2,.1)
			tmp = [np.var(disc_payoffs['ArtMean'] + c*(disc_payoffs['GeoMean'] - self.truePrice[type])) for c in corr]

			plt.plot(corr, tmp)
			plt.savefig('variance_' + type + '.png')

			corr = corr[tmp.index(min(tmp))]
			tmp = disc_payoffs['ArtMean'] + corr*(disc_payoffs['GeoMean'] - self.truePrice[type])

			sol[type] = conf_int(tmp, self.conf_lvl)
			sol[type]['corr'] = -corr

		return sol

def asian_callMC(S0, T, K, r, sigma, M, n):
	model = asian_MC_model(S0, T, K, r, sigma, M, n)
	return model.run()

res = asian_callMC(**params)
print(res)

# test na minimalne n, dla którego oba wyniki sá jednakowe
while round(abs(res['Cont']['Price'] - res['Disc']['Price']),4) != 0. and params['n'] < 10**6:
	print(params['n'], round(abs(res['Cont']['Price'] - res['Disc']['Price']),4))
	params['n'] *= 2
	res = asian_callMC(**params)
print(params['n'])