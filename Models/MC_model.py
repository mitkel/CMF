import numpy as np
import numpy.random as npr
from scipy.stats import norm
from auxiliary_functions import conf_int, zero_ceil

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

########################################
############### MODELS #################
########################################

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
			corr = corr[tmp.index(min(tmp))]
			tmp = disc_payoffs['ArtMean'] + corr*(disc_payoffs['GeoMean'] - self.truePrice[type])

			sol[type] = conf_int(tmp, self.conf_lvl)
			sol[type]['corr'] = -corr

		return sol

# tests
# params3 = {'r': .05, 'sigma': .3, 'K': 50., 'S0': 50., 'T':.5, 'M': 10**4, 'n':10**3}
# model = asian_MC_model(**params3)
# print(model.run())


class Heston_MC_model(MC_model):
	def __init__(self, S0, V0, T, K, r, a, b, sigma, rho, M, n, CallPut = 'Call', conf_lvl = .95):
		super(Heston_MC_model, self).__init__(S0, T, n, r, M, K, CallPut)
		self.state = {'S':np.array([S0]*self.SimNo),
					  'V':np.array([V0]*self.SimNo),
					  'Z':np.log(np.array([S0]*self.SimNo)),
					  't':0}
		self.a = a
		self.b = b
		self.sigma = sigma
		self.corr = rho
		self.conf_lvl = conf_lvl

	def Finish(self):
		sol = {}
		for type in ['S','Z']:
			if type == 'Z':
				tmp = np.exp(self.state[type])
			else:
				tmp = self.state[type]

			disc_payoffs = np.exp(-self.r*self.state['t'])*self.payment(tmp)
			sol[type] = conf_int(disc_payoffs, self.conf_lvl)

		return sol

	def goFwd(self):
		randIncr = npr.multivariate_normal([0,0], [[self.stepSize, self.corr*self.stepSize],
												   [self.corr*self.stepSize, self.stepSize]],
										   size=self.SimNo)

		self.StockStep(np.array([x[0] for x in randIncr]))
		self.logStep(np.array([x[0] for x in randIncr]))
		self.VolStep(np.array([x[1] for x in randIncr]))
		self.state['t'] += self.stepSize

	def StockStep(self, randIncr):
		self.state['S'] += self.state['S']*(self.r*self.stepSize +
											np.sqrt(zero_ceil(self.state['V']))*randIncr)

	def logStep(self, randIncr):
		self.state['Z'] += (self.r-.5*self.state['V'])*self.stepSize + \
						   np.sqrt(zero_ceil(self.state['V']))*randIncr

	def VolStep(self, randIncr):
		self.state['V'] += self.a*(self.b-self.state['V'])*self.stepSize +\
						   self.sigma*np.sqrt(zero_ceil(self.state['V']))*randIncr +\
						   .25*self.sigma**2*(randIncr**2-self.stepSize)

# tests
# params1 = {'r': .05, 'sigma': .4, 'K': 45, 'V0': .06, 'S0': 50., 'T':2, 'a': 2, 'b': .04, 'rho': -.7, 'M': 10**3, 'n':4*10**2}
# model = Heston_MC_model(**params1, CallPut='Call')
# print(model.run())
# params2 = {'r': .05, 'sigma': 1., 'K': 100, 'V0': .09, 'S0': 100., 'T':5, 'a': 2, 'b': .09, 'rho': -.3, 'M': 10**3, 'n':4*10**2}
# model = Heston_MC_model(**params2, CallPut='Call')
# print(model.run())
