import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from scipy import stats

NSIMS = 1000
df = pd.read_excel("century.xlsx", sheet_name = 'price')
vol = df["Volatility"].values[1:]
NDISPLAYS = 5
N = len(vol)
L = 9
price = df['Price'].values
dividend = df['Dividends'].values[1:]
dfEarnings = pd.read_excel('century.xlsx', sheet_name = 'earnings')
earnings = dfEarnings['Earnings'].values
cpi = dfEarnings['CPI'].values
inflation = np.diff(np.log(cpi))
window = 5 # change if needed to any number between 1 and 10
inflMode = 'Real' # change if needed between 'Nominal' and 'Real'
ABC = {'Volatility' : 10, 'BAA-AAA' : 2, 'AAA-Long' : 2, 'Long-Short' : 2, 'Index' : 1000} # initial conditions

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

def gaussNoise(data, size):
    covMatrix = data.cov()
    simNoise = []
    dim = len(data.keys())
    for sim in range(NSIMS):
        simNoise.append(np.transpose(np.random.multivariate_normal(np.zeros(dim), covMatrix, size)))
    return simNoise

def KDE(data, size):
    method = stats.gaussian_kde(np.transpose(data.values), bw_method = 'silverman')
    simNoise = []
    for sim in range(NSIMS):
        simNoise.append(np.array(method.resample(size)))
    return simNoise

lvol = np.log(vol)
RegVol = stats.linregress(lvol[:-1], lvol[1:])
betaVol = RegVol.slope
alphaVol = RegVol.intercept
residVol = np.array([lvol[k+1] - betaVol * lvol[k] - alphaVol for k in range(N-1)])
meanVol = np.mean(vol)
spreads = {'BAA-AAA' : (df['BAA'] - df['AAA']).values, 'AAA-Long' : (df['AAA'] - df['Long']).values, 'Long-Short' : (df['Long'] - df['Short']).values}
DFreg = pd.DataFrame({'const' : 1/vol, 'vol' : 1})
latest = {'Volatility' : round(vol[-1], 1), 'Index' : round(index[-1], 3)}
means = {'Volatility' : round(meanVol, 1), 'Index' : round(cumearn[-1]/np.mean(earnyield), 3)}
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
RegGrowth = OLS(growth/vol, DFreg).fit()
allModels['Growth'] = RegGrowth
resgrowth = RegGrowth.resid
allResids['Growth'] = resgrowth.values[1:]
DFfull = DFreg 
DFfull['EarningsYield'] = np.log(earnyield[:-1])/vol

for key in nrets:
    returns = nrets[key]
    Regression = OLS(returns, DFfull).fit()
    allModels[key] = Regression
    res = Regression.resid
    allResids[key] = res.values[1:]
        
def simReturns(initial, residMode, horizon):
    if residMode == 'Gauss':
        innovations = gaussNoise(allResids, horizon)
    if residMode == 'KDE':
        innovations = KDE(allResids, horizon)
    allSims = []
    for sim in range(NSIMS):
        simRet = []
        simLVol = [np.log(initial['Volatility'])]
        innov = innovations[sim]
        for t in range(horizon):
            simLVol.append(simLVol[-1]*betaVol + alphaVol + innov[0, t])
        simVol = np.exp(simLVol)
        simFactors = dict({})
        for key in spreads:
            simFactors[key] = [initial[key]]
        simFactors['Growth'] = []
        for t in range(horizon):
            currentSpreadFactors = [1/simVol[t+1], 1]
            for key in spreads:
                currentSpreadFactors.append(simFactors[key][t]/simVol[t+1])
            currentInnov = {'BAA-AAA' : innov[1, t], 'AAA-Long' : innov[2, t], 'Long-Short' :innov[3, t],'Growth': innov[4, t]}
            for key in simFactors:
                simFactors[key].append((allModels[key].predict(currentSpreadFactors)[0] + currentInnov[key]) * simVol[t+1])
        simGrowth = simFactors['Growth']
        oldEarn = earn[-window:]
        initialCAEY = np.mean(oldEarn)/initial['Index']
        simEarn = np.append(oldEarn, oldEarn[-1] * np.exp(np.cumsum(simGrowth)))
        simPrice = [initial['Index']]
        simCAEY = [initialCAEY]
        current = {'Volatility' : round(simVol[0], 1), 'EarningsYield' : round(initialCAEY, 3)}
        simRet = []
        for t in range(horizon):
            allFactors = [1/simVol[t+1], 1]
            for key in spreads:
                allFactors.append(simFactors[key][t]/simVol[t+1])
            allFactors.append(np.log(simCAEY[-1])/simVol[t+1])
            simPriceRet = simVol[t+1] * (allModels['Price'].predict(allFactors)[0] + innov[5, t])
            simPrice.append(simPrice[t] * np.exp(simPriceRet))
            simCAEY.append(np.mean(simEarn[t+1:t+1+window])/simPrice[t+1])
            simTotalRet = simVol[t+1] * (allModels['Total'].predict(allFactors)[0] + innov[6, t])
            simRet.append(simTotalRet)
        allSims.append(simRet)
    return np.array(allSims)

def output(initial, residMode, horizon, initialWealth, flow):
    if flow == 0:
        flowText = 'No regular contributions or withdrawals'
    if flow > 0:
        flowText = 'Contributions ' + str(flow) + ' per year'
    if flow < 0:
        flowText = 'Withdrawals ' + str(abs(flow)) + ' per year'
    paths = []
    timeAvgRets = []
    simulatedReturns = simReturns(initial, residMode, horizon)
    for sim in range(NSIMS):
        path = [initialWealth]
        simReturn = simulatedReturns[sim]
        timeAvgRets.append(np.mean(simReturn))
        for t in range(horizon):
            if (path[t] == 0):
                path.append(0)
            else:
                new = max(path[t] * np.exp(simReturn[t]) + flow, 0)
                path.append(new)
        paths.append(path)
    paths = np.array(paths)
    avgRet = np.mean([timeAvgRets[sim] for sim in range(NSIMS) if paths[sim, -1] > 0])
    wealthMean = np.mean(paths[:, -1])
    meanProb = np.mean([paths[sim, -1] > wealthMean for sim in range(NSIMS)])
    ruinProb = np.mean([paths[sim, -1] == 0 for sim in range(NSIMS)])
    sortedIndices = np.argsort(paths[:, -1])
    selectedIndices = [sortedIndices[int(NSIMS*(2*k+1)/(2*NDISPLAYS))] for k in range(NDISPLAYS)]
    times = range(horizon + 1)
    simText = str(NSIMS) + ' of simulations'
    timeHorizonText = 'Time Horizon: ' + str(horizon) + ' years'
    inflText = inflMode + ' returns'
    initWealthText = 'Initial Wealth ' + str(round(initialWealth))
    Portfolio = 'The portfolio: 100% Large Stocks'
    modelText = 'Volatility, 3 spreads, trailing earnings yield for ' + str(window) + ' years'
    initMarketText = 'Initial (Current) conditions: ' + ' '.join([key + ' ' + str(initial[key]) for key in initial])
    avgMarketText = 'Historical averages: ' + ' '.join([key + ' ' + str(means[key]) for key in means])
    SetupText = 'SETUP: ' + simText + '\n' + modelText + '\n' + Portfolio + '\n' + timeHorizonText + '\n' + inflText + '\n' + initWealthText +'\n' + initMarketText + '\n' + avgMarketText + '\n' + flowText + '\n'
    if np.isnan(avgRet):
        ResultText = 'RESULTS: 100% Ruin Probability, always zero wealth'
    else:
        RuinProbText = str(round(100*ruinProb, 2)) + '% Ruin Probability'
        AvgRetText = 'time averaged annual returns:\naverage over all paths without ruin ' + str(round(100*avgRet, 2)) + '%'
        MeanText = 'average final wealth ' + str(round(wealthMean))
        MeanCompText = 'final wealth exceeds average with probability ' + str(round(100*meanProb, 2)) + '%'
        ResultText = 'RESULTS: ' + RuinProbText + '\n' + AvgRetText + '\n' + MeanText + '\n' + MeanCompText
    bigTitle = SetupText + '\n' + ResultText + '\n'
    plt.plot([0, 0], color = 'w', label = bigTitle)
    for display in range(NDISPLAYS):
        index = selectedIndices[display]
        rankText = ' final wealth, ranked ' + str(round(100*(2*display + 1)/(2*NDISPLAYS))) + '% '
        selectTerminalWealth = round(paths[index, -1])
        if (selectTerminalWealth == 0):
            plt.plot(times, paths[index], label = '0' + rankText + 'Gone Bust !!!')
        else:
            plt.plot(times, paths[index], label = str(selectTerminalWealth) + rankText + 'returns: ' + str(round(100*timeAvgRets[index], 2)) + '%')
    plt.xlabel('Years')
    plt.ylabel('Wealth')
    plt.title('Wealth Plot')
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', prop={'size': 12})
    image_path = 'wealth.png'
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

output(ABC, 'Gauss', 30, 1000, -40)