from math import log, floor, ceil, fmod, sqrt, sin, cos, pi, exp
import numpy as np
import numpy.random as npr
from scipy.stats import norm
from pprint import pprint as print
from CW2_data import params

def halton(dim, nbpts):
    h = np.empty(nbpts * dim)
    h.fill(np.nan)
    p = np.empty(nbpts)
    p.fill(np.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = log(nbpts + 1)
    for i in range(dim):
        b = P[i]    # prime used
        n = int(ceil(lognbpts / log(b))) #n: b**n > nbpts

        for t in range(n):
            p[t] = pow(b, -(t + 1) ) #[0,1]_b base

        for j in range(nbpts):
            d = j + 1
            sum_ = fmod(d, b) * p[0]
            for t in range(1, n):
                d = floor(d / b)
                sum_ += fmod(d, b) * p[t]

            h[j*dim + i] = sum_

    return h.reshape(nbpts, dim) 

def BoxMuller(U,V):
    R = np.sqrt(-2*log(U))
    theta = 2*pi*V
    return R*cos(theta), R*sin(theta)

def normal_sequence(U):
    res = [BoxMuller(*u) for u in U]
    return res

def ST(S0, r, sigma, T, Z):
    return S0*np.exp((r-.5*sigma**2)*T + sigma*sqrt(T)*np.array(Z))

def OptionOut(S, K):
    return [max(K-s, 0) for s in S]

def true_price(S0, r, sigma, T, K):
    d1 = (log(S0/K) + r*T + .5*T*sigma**2)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    price = -S0*norm.cdf(-d1)+exp(-r*T)*K*norm.cdf(-d2)
    return price

def PutMC(S0, r, sigma, T, K, M, confidence = .95):
    M = int(2*ceil(M/2))
    RND = npr.uniform(size=M)
    RND_est = 0
    RND = [OptionOut(ST(S0, r, sigma, T, BoxMuller(*u)), K) for u in RND.reshape(int(M/2), 2)]
    RND_est = exp(-r*T)*np.mean(RND)
    RND_var = sqrt(np.var(RND))
    h = 1-norm.cdf((confidence+1)/2)
    RND_int = [RND_est-h, RND_est+h]


    RND = normal_sequence(halton(2, int(M/2)))
    QMC_est = 0
    QMC_est = exp(-r*T)*np.mean([OptionOut(ST(S0, r, sigma, T, u), K) for u in RND])

    price = true_price(S0, r, sigma, T, K)

    return {'MC':RND_est, 'QMC':QMC_est, 'MCerr':abs(RND_est-price), 'QMCerr':abs(QMC_est-price), \
            'price':price, 'confInt': RND_int}

print(PutMC(**params))