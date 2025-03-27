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

df = pd.read_excel("century.xlsx", sheet_name = 'price')
vol = df["Volatility"].values[1:]
N = len(vol)
L = 9
price = df['Price'].values
dividend = df['Dividends'].values[1:]
dfEarnings = pd.read_excel('century.xlsx', sheet_name = 'earnings')
earnings = dfEarnings['Earnings'].values
cpi = dfEarnings['CPI'].values
inflation = np.diff(np.log(cpi))
window = 10 # change if needed to any number between 1 and 10
inflMode = 'Real' # change if needed between 'Nominal' and 'Real'
if inflMode == 'Nominal':
    index = price
    div = dividend
    total = np.array([np.log(index[k+1] + dividend[k]) - np.log(index[k]) for k in range(N)])
    earn = earnings
if inflMode == 'Real':
    index = cpi[-1]*price/cpi[L:]
    div = cpi[-1]*dividend/cpi[L+1:]
    total = np.array([np.log(index[k+1] + div[k]) - np.log(index[k]) for k in range(N)])
    earn = cpi[-1]*earnings/cpi
Nprice = np.diff(np.log(index))/vol
Ntotal = total/vol
growth = np.diff(np.log(earn))[L:]
cumearn = np.array([np.mean(earn[k-window:k]) for k in range(L + 1, L + N + 2)])
earnyield = cumearn/index
plt.plot(range(1928, 1929 + N), earnyield)
plt.title('Cyclically Adjusted Earnings Yield Averaged over ' + str(window) + ' Years')
plt.savefig('earnings-yield.png')
plt.close()
lvol = np.log(vol)
RegVol = stats.linregress(lvol[:-1], lvol[1:])
betaVol = RegVol.slope
alphaVol = RegVol.intercept
residVol = [lvol[k+1] - betaVol * lvol[k] - alphaVol for k in range(N-1)]
meanVol = np.mean(vol)
spreads = {}
spreads['BAA-AAA'] = (df['BAA'] - df['AAA']).values
spreads['AAA-Long'] = (df['AAA'] - df['Long']).values
spreads['Long-Short'] = (df['Long'] - df['Short']).values
DFreg = pd.DataFrame({'const' : 1/vol, 'vol' : 1})
for key in spreads:
    DFreg[key] = spreads[key][:-1]/vol

for key in spreads:
    print('Regression of', key)
    Reg = OLS(list(np.diff(spreads[key])/vol), DFreg).fit()
    print(Reg.summary())
    resid = Reg.resid
    plots(resid, key)
    print(analysis(resid, key))
    
nrets = {'price' : Nprice, 'total' : Ntotal}
RegGrowth = OLS(growth, DFreg).fit()
print('Earnings Growth')
print(RegGrowth.summary())
resgrowth = RegGrowth.resid
plots(resgrowth, 'Growth')
print(analysis(resgrowth, 'Growth'))
DFfull = DFreg 
DFfull['EarningsYield'] = np.log(earnyield[:-1])/vol

for key in nrets:
    print('Regression for Returns', key, '\n\n')
    returns = nrets[key]
    Regression = OLS(returns, DFfull).fit()
    print(Regression.summary())
    res = Regression.resid
    plots(res, key + ' Returns Full')
    print(analysis(res, key + ' Returns Full'))