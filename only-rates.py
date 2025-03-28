import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
from statsmodels.tsa.stattools import acf

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    
def analysis(data, label):
    skew = round(stats.skew(data), 3)
    kurt = round(stats.kurtosis(data), 3)
    SWp = round(stats.shapiro(data)[1], 3)
    JBp = round(stats.jarque_bera(data)[1], 3)
    L1orig = round(sum(abs(acf(data, nlags = 5)[1:])), 1)
    L1abs = round(sum(abs(acf(abs(data), nlags = 5)[1:])), 3)
    print('skewness, kurtosis, Shapiro-Wilk p, Jarque-Bera p, L1 for 5 ACF of Z and of |Z|')
    return [label, skew, kurt, SWp, JBp, L1orig, L1abs]

def gaussNoise(data, size):
    covMatrix = data.cov()
    dim = len(data.keys())
    simNoise = np.transpose(np.random.multivariate_normal(np.zeros(dim), covMatrix, size))
    return simNoise

def KDE(data, size):
    method = stats.gaussian_kde(np.transpose(data.values), bw_method = 'silverman')
    simNoise = np.array(method.resample(size))
    return simNoise

df = pd.read_excel("century.xlsx", sheet_name = 'price')
vol = df["Volatility"].values[1:]
NDISPLAYS = 5
N = len(vol)
L = 9

price = df['Price'].values
dividend = df['Dividends'].values[1:]
dfEarnings = pd.read_excel('century.xlsx', sheet_name = 'earnings')
cpi = dfEarnings['CPI'].values

window = 5 # change if needed to any number between 1 and 10
inflMode = 'Real' # change if needed between 'Nominal' and 'Real'
residMode = 'Gauss' # change if want Kernel Density

if inflMode == 'Nominal':
    index = price
    div = dividend
            
if inflMode == 'Real':
    index = cpi[-1]*price/cpi[L:]
    div = cpi[-1]*dividend/cpi[L+1:]
    
total = np.array([np.log(index[k+1] + div[k]) - np.log(index[k]) for k in range(N)])
Nprice = np.diff(np.log(index))/vol
Ntotal = total/vol

lvol = np.log(vol)
RegVol = stats.linregress(lvol[:-1], lvol[1:])
betaVol = RegVol.slope
alphaVol = RegVol.intercept
residVol = np.array([lvol[k+1] - betaVol * lvol[k] - alphaVol for k in range(N-1)])
meanVol = np.mean(vol)

spreads = {'BAA-AAA' : (df['BAA'] - df['AAA']).values, 'AAA-Long' : (df['AAA'] - df['Long']).values, 'Long-Short' : (df['Long'] - df['Short']).values}
DFreg = pd.DataFrame({'const' : 1/vol, 'vol' : 1})
latest = {'Volatility' : round(vol[-1], 1)}
means = {'Volatility' : round(meanVol, 2)}

for key in spreads:
    DFreg[key] = spreads[key][:-1]/vol
    latest[key] = round(spreads[key][-1], 2)
    means[key] = round(np.mean(spreads[key]), 2)

allResids = pd.DataFrame({'Volatility' : residVol})
allModels = dict({})

for key in spreads:
    Reg = OLS(spreads[key][1:]/vol, DFreg).fit()
    resid = Reg.resid
    allModels[key] = Reg
    allResids[key] = resid.values[1:]
    
nrets = {'Price' : Nprice, 'Total' : Ntotal}

for key in nrets:
    returns = nrets[key]
    Regression = OLS(returns, DFreg).fit()
    allModels[key] = Regression
    print('Regression for returns ', key)
    print(Regression.summary())
    res = Regression.resid
    allResids[key] = res.values[1:]
    plots(res, key)
    print(analysis(res, key))

horizon = 20 # change if necessary
initial = means # change of necessary

if residMode == 'Gauss':
    innov = gaussNoise(allResids, horizon)
if residMode == 'KDE':
    innov = KDE(allResids, horizon)

simLVol = [np.log(initial['Volatility'])]
for t in range(horizon):
    simLVol.append(simLVol[-1]*betaVol + alphaVol + innov[0, t])
simVol = np.exp(simLVol)
simFactors = dict({})
for key in spreads:
    simFactors[key] = [initial[key]]
simPriceRet = []
simTotalRet = []
for t in range(horizon):
    currentSpreadFactors = [1/simVol[t+1], 1]
    for key in spreads:
        currentSpreadFactors.append(simFactors[key][t]/simVol[t+1])
        currentInnov = {'BAA-AAA' : innov[1, t], 'AAA-Long' : innov[2, t], 'Long-Short' : innov[3, t]}
    for key in simFactors:
        simFactors[key].append((allModels[key].predict(currentSpreadFactors)[0] + currentInnov[key]) * simVol[t+1])
    simPriceRet.append((allModels['Price'].predict(currentSpreadFactors)[0] + innov[4, t]) * simVol[t+1])
    simTotalRet.append((allModels['Total'].predict(currentSpreadFactors)[0] + innov[5, t]) * simVol[t+1])
simPrice = np.exp(np.append(np.array([0]), np.cumsum(simPriceRet)))
simTotal = np.exp(np.append(np.array([0]), np.cumsum(simTotalRet)))
for key in simFactors:
    plt.plot(simFactors[key], label = key)
plt.title('Spreads')
plt.legend()
plt.show()
plt.plot(simVol)
plt.title('Simulated Volatility')
plt.show()
plt.plot(simPrice)
plt.plot(simTotal)
plt.title('Simulated Price and Wealth')
plt.show()