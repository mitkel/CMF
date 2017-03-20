from pprint import pprint as print
import numpy as np

import CW1_data

params = CW1_data.params


class CEV_european_model():
    def __init__(self, M, K, S0, T, r, sigma, CallPut, beta):
        self.stepsNo = M
        self.strike = K
        self.S0 = S0
        self.sigma = sigma
        self.r = r
        self.maturity = T
        self.type = CallPut

        self.beta = beta
        self.alpha = 1 - beta / 2
        self.delta = self.sigma * S0 ** self.alpha

        self.stepSize = T / M
        self.increment = np.sqrt(self.stepSize)
        self.state = {'X': np.array(1 / (self.alpha * self.sigma)),
                      'S': np.array(S0),
                      'V': np.array(None)}

    def payment(self, S):
        phi = ((self.type == 'Put') - .5) * 2
        return max(0, phi * (self.strike - S))

    def goFwd(self):
        self.state['X'] = np.append(np.atleast_1d(self.state['X']) - self.increment,
                                    np.atleast_1d(self.state['X'])[-1] + self.increment)
        self.state['S'] = (self.state['X'] * self.alpha * self.delta) ** (1 / self.alpha)

    def pUP(self, state_act, state_ahead):
        nom = [y * np.exp(self.r * self.stepSize) - x for x, y in zip(state_ahead[:-1], state_act)]
        den = [x - y for x, y in zip(state_ahead[1:], state_ahead[:-1])]

        return np.array([(x > 0) * (y > 0) * min(y, 1) for x, y in zip(self.state['S'], np.divide(nom, den))])

    def goBwd(self):
        oldS = self.state['S'].copy()

        self.state['X'] = np.atleast_1d(self.state['X'])[:-1] + self.increment
        self.state['S'] = (self.state['X'] * self.alpha * self.delta) ** (1 / self.alpha)

        prob = self.pUP(self.state['S'], oldS)
        self.state['V'] = (np.multiply(np.atleast_1d(self.state['V'])[1:], prob) +
                           np.multiply(np.atleast_1d(self.state['V'])[:-1], (1 - prob))) * \
                          np.exp(- self.stepSize * self.r)

    def run(self):
        for _ in range(self.stepsNo + 2):
            self.goFwd()
            yield (self.state)

        self.state['V'] = np.array([self.payment(x) for x in self.state['S']])
        yield (self.state)

        for _ in range(self.stepsNo + 2):
            self.goBwd()
            yield (self.state)

    def analyze(self):
        V = [None] * 3
        S = [None] * 3
        W = [None] * 2
        for self.state in self.run():
            if self.state['X'][0] is not None:
                if len(np.atleast_1d(self.state['V'])) == 5:
                    W[1] = self.state['V'][2]
                elif len(np.atleast_1d(self.state['V'])) == 3:
                    V = self.state['V'].copy()
                    S = self.state['S'].copy()
                elif len(np.atleast_1d(self.state['V'])) == 1:
                    W[0] = np.atleast_1d(self.state['V'])[0]
        Price = V[1]
        Delta = (V[2] - V[0]) / (S[2] - S[0])
        Gamma = 2 * ((V[2] - V[1]) / (S[2] - S[1]) - (V[1] - V[0]) / (S[1] - S[0])) / (S[2] - S[0])
        Theta = (W[1] - W[0]) / (4 * self.stepSize)
        return {'Price': Price, 'Delta': Delta, 'Gamma': Gamma, 'Theta': Theta}


def binomial_CEV(**kwargs):
    model = CEV_european_model(**kwargs)
    print(model.analyze())

    # import numpy as np
    # import matplotlib.pyplot as plt
    # greka = ['Theta', 'Delta', 'Gamma','Price']
    # res = {'Theta': [], 'Delta': [], 'Gamma': [],'Price':[]}

    # T = 100+np.arange(50)*100
    # for j in T:
    #     params['M'] = j
    #     model = CEV_european_model(CallPut='Call', **params)
    #     tmp = model.analyze()
    #     for name in greka:
    #         res[name] = np.append(res[name], tmp[name])
    #
    # for name in greka:
    #     plt.plot(T, res[name], color="blue", linestyle="-", label = name)
    #     plt.savefig(name+ '_bigSteps.png')
    #     plt.clf()
    #
    # T = 5+np.arange(196)
    # for j in T:
    #     params['M'] = j
    #     model = CEV_european_model(CallPut='Call', **params)
    #     tmp = model.analyze()
    #     for name in greka:
    #         res[name] = np.append(res[name], tmp[name])
    #
    # for name in greka:
    #     plt.plot(T, res[name], color="blue", linestyle="-", label = name)
    #     plt.plot(T[::2], res[name][::2], color="orange", linestyle="--", label = "even")
    #     plt.plot(T[1::2], res[name][1::2], color="green", linestyle="--", label = "odd")
    #     plt.savefig(name+ '_smallSteps.png')
    #     plt.clf()
    #
    # T = 5000+np.arange(6)*1000
    # for j in T:
    #     params['M'] = j
    #     model = CEV_european_model(CallPut='Call', **params)
    #     tmp = model.analyze()
    #     for name in greka:
    #         res[name] = np.append(res[name], tmp[name])
    #
    # for name in greka:
    #     plt.plot(T, res[name], color="blue", linestyle="-", label = name)
    #     plt.savefig(name+ '_hugeSteps.png')
    #     plt.clf()
