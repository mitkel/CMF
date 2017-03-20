import numpy as np
import matplotlib.pyplot as plt

params = {'r': 0.09, 'M': 100, 'sigma': 0.25, 'beta': 1, 'K': 100, 'T': 1, 'S0': 105}

greek = ['Theta', 'Delta', 'Gamma','Price']
res = {'Theta': [], 'Delta': [], 'Gamma': [],'Price':[]}

T = 100+np.arange(50)*100
for j in T:
    params['M'] = j
    model = CEV_european_model(CallPut='Call', **params)
    tmp = model.analyze()
    for name in greka:
        res[name] = np.append(res[name], tmp[name])

for name in greka:
    plt.plot(T, res[name], color="blue", linestyle="-", label = name)
    plt.savefig(name+ '_bigSteps.png')
    plt.clf()

T = 5+np.arange(196)
for j in T:
    params['M'] = j
    model = CEV_european_model(CallPut='Call', **params)
    tmp = model.analyze()
    for name in greka:
        res[name] = np.append(res[name], tmp[name])

for name in greka:
    plt.plot(T, res[name], color="blue", linestyle="-", label = name)
    plt.plot(T[::2], res[name][::2], color="orange", linestyle="--", label = "even")
    plt.plot(T[1::2], res[name][1::2], color="green", linestyle="--", label = "odd")
    plt.savefig(name+ '_smallSteps.png')
    plt.clf()

T = 5000+np.arange(6)*1000
for j in T:
    params['M'] = j
    model = CEV_european_model(CallPut='Call', **params)
    tmp = model.analyze()
    for name in greka:
        res[name] = np.append(res[name], tmp[name])

for name in greka:
    plt.plot(T, res[name], color="blue", linestyle="-", label = name)
    plt.savefig(name+ '_hugeSteps.png')
    plt.clf()
