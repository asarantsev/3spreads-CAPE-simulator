import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from sklearn.impute import KNNImputer

skewAll = []
kurtAll = []
SWp = []
JBp = []
L1O = []
L1A = []

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
    
def analysis(data):
    skewAll.append(round(stats.skew(data), 3))
    kurtAll.append(round(stats.kurtosis(data), 3))
    SWp.append(round(stats.shapiro(data)[1], 3))
    JBp.append(round(stats.jarque_bera(data)[1], 3))
    L1O.append(round(sum(abs(acf(data, nlags = 5)[1:])), 3))
    L1A.append(round(sum(abs(acf(abs(data), nlags = 5)[1:])), 3))
    
DF = pd.read_excel('overall.xlsx', sheet_name = None)
dfPrice = DF['main']
vol = dfPrice['Volatility'].values[1:]
N = len(vol)
price = dfPrice['Price'].values
dividend = dfPrice['Dividends'].values[1:]
baa = dfPrice['BAA'].values
spread = dfPrice['Long'].values - dfPrice['Short'].values

plt.plot(range(1928, 1929 + N), baa)
plt.title('BAA')
plt.savefig('baa.png')
plt.close()

plt.plot(range(1928, 1929 + N), spread)
plt.title('Spread')
plt.savefig('spread.png')
plt.close()

dfEarnings = DF['earnings']
earnings = dfEarnings['Earnings'].values
lvol = np.log(vol)
total = np.array([np.log(price[k+1] + dividend[k]) - np.log(price[k]) for k in range(N)])
earn = earnings
nUSAret = 100*total/vol
L = 9
earngr = np.diff(np.log(earn[L:]))
ngrowth = 100*earngr/vol

RegVol = OLS(lvol[1:], pd.DataFrame({'const' : 1, 'lag' : lvol[:-1]})).fit()
DFrates = pd.DataFrame({'const' : 1, 'spread' : spread[:-1], 'baa' : np.log(baa)[:-1]})
RegSpread = OLS(np.diff(spread), DFrates).fit()
RegBAA = OLS(np.diff(np.log(baa)), DFrates).fit()
GrowthDF = pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'spread' : spread[:-1]/vol, 'baa' : np.diff(baa)/vol})
RegGrowth = OLS(ngrowth, GrowthDF).fit()

world = DF['world'] 
intlReturns = world['International'].values # international returns
NINTL = len(intlReturns)
nIntlRet = 100 * np.log(1 + intlReturns)/vol[-NINTL:] # normalized intl returns in %

bonds = DF['bonds']
wealthBond = bonds['Bond Wealth'].values
NBOND = len(wealthBond) - 1
bondRet = np.diff(np.log(wealthBond)) # bond returns in %

dfBAA = pd.DataFrame({'const' : 1, 'dur' : np.diff(baa)})
RegBond = OLS(100*bondRet - baa[-NBOND-1:-1], dfBAA.iloc[-NBOND:]).fit()

window = 10
cumearn = np.array([np.mean(earn[k-window:k]) for k in range(L + 1, L + N + 2)])
IDY = total - np.diff(np.log(cumearn))
cumIDY = np.append(np.array([0]), np.cumsum(IDY))
AllFactors = pd.DataFrame({'const' : 1, 'trend' : range(N), 'Bubble' : -cumIDY[:-1]})
RegVal = OLS(IDY, AllFactors).fit()
Valuation = cumIDY - np.array(range(N+1)) * (RegVal.params['trend'] / RegVal.params['Bubble'])
plt.plot(range(1928, 1929 + N), Valuation)
plt.title('New Valuation Measure')
plt.savefig('NewMeasure.png')
plt.close()

regDF = pd.DataFrame({'const' : 1/vol, 'duration' : np.diff(baa)/vol, 'vol' : 1, 'Valuation' : Valuation[:-1]/vol, 'spread' : spread[:-1]/vol}) 
RegUSA = OLS(nUSAret, regDF).fit()
RegIntl = OLS(nIntlRet, regDF[['const', 'duration', 'vol', 'Valuation']].iloc[-NINTL:]).fit()

allRegs = [RegVol, RegSpread, RegBAA, RegGrowth, RegVal, RegUSA, RegIntl, RegBond]
allNames = ['vol', 'spread', 'baa-rate', 'earn-growth', 'bubble', 'usa-ret', 'intl-ret', 'bond-ret']
allResiduals = pd.DataFrame(columns = allNames)
DIM = 8    
lengths = []

for k in range(DIM):
    print(allNames[k], '\n') # name of regression
    regression = allRegs[k] # regression itself
    print(regression.summary()) # print regression summary
    print('coefficients')
    print(regression.params) # print regression parameters
    resids = regression.resid.values # residuals of this regression
    lengths.append(len(resids))
    allResiduals[allNames[k]] = np.pad(resids[::-1], (0, N - lengths[k]), constant_values = np.nan)
    plots(resids, allNames[k]) # normality and autocorrelation function plots
    analysis(resids) # are these residuals normal white noise?

cutResidDF = allResiduals.drop(['intl-ret', 'bond-ret'], axis = 1)
cutResidDF['vol'].iloc[-1] = 0
covMatrix = allResiduals.cov()
corrMatrix = allResiduals.corr()
statDF = pd.DataFrame({'reg' : allNames, 'skew': skewAll, 'kurt' : kurtAll, 'SW' : SWp, 'JB' : JBp, 'L1O': L1O, 'L1A' : L1A, 'length' : lengths})
print(statDF)

pMatrix = [[0 for item in range(DIM)] for item in range(DIM)]
for k in range(DIM):
    for l in range(DIM):
        if k == l:
            pMatrix[k][k] = 1
        else:
            x = allResiduals[allNames[k]]
            y = allResiduals[allNames[l]]
            Q = min(lengths[k], lengths[l])
            pMatrix[k][l] = round(stats.pearsonr(x[:Q], y[:Q])[1], 2)*100
print('empirical matrix')
for k in range(DIM):
    print(pMatrix[k])