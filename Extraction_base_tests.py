import yfinance as yf
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use("Qt5Agg")  # backend interactif
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.stats import spearmanr, pearsonr





# 1) Récupérer données (exemple daily, 10 ans)
tickers = {'gold':'GC=F', 'silver':'SI=F'} #Ici changer ticker pour celui/ceux qu'on veut
data = {}
for name,t in tickers.items():
    df = yf.download(t, period='20y', interval='1d', progress=False)
    df = df['Close'].dropna()
    data[name] = df

gold = data['gold']
silver = data['silver']

# 2) Log-prices et log-returns
logp_gold = np.log(gold)

r_gold = logp_gold.diff().iloc[:, 0].dropna()
r_r_gold=gold.pct_change().iloc[:,0].dropna()

logp_silver = np.log(silver)
r_silver = logp_silver.diff().iloc[:, 0].dropna()
r_r_silver=silver.pct_change().iloc[:,0].dropna()

# 3) EDA simple
plt.figure(figsize=(10,4))
plt.plot(gold.index, gold)
plt.title('Gold (Close)')
plt.show()
plt.figure(figsize=(10,4))
plt.plot(silver.index, silver)
plt.title('Silver (Close)')
plt.show()

plt.figure(figsize=(10,4))

r_gold.hist(bins=200)
plt.title('Histogramme rendements log Gold')
plt.show()
plt.figure(figsize=(10,4))
r_silver.hist(bins=200)
plt.title('Histogramme rendements log Silver')
plt.show()

# 4) Test de stationnarité sur rendements
result=adfuller(r_gold)  # p-value
print("ADF statistic:", result[0])
print("p-value:", result[1])
print("Critical values:", result[4])
#5) QQ plots

import scipy.stats as stats
plt.figure()

stats.probplot(r_gold, dist="norm", plot=plt)
plt.title("QQ-plot des log-rendements de l'or")
plt.show()

plt.figure()

stats.probplot(r_silver, dist="norm", plot=plt)
plt.title("QQ-plot des log-rendements de l'argent")
plt.show()
# 6) Autocorreclations

from statsmodels.graphics.tsaplots import plot_acf

# ACF des rendements
plot_acf(r_gold, lags=40)
plt.title("ACF des log-rendements")
plt.show()

# ACF des rendements au carré
plot_acf(r_gold**2, lags=40)
plt.title("ACF des log-rendements² (volatilité)")
plt.show()
# ACF des rendements
plot_acf(r_silver, lags=40)
plt.title("ACF des log-rendements")
plt.show()

# ACF des rendements au carré
plot_acf(r_silver**2, lags=40)
plt.title("ACF des log-rendements² (volatilité)")
plt.show()

mean_gold=r_gold.mean()*252


mean_silver=r_silver.mean()*252
std_gold=r_gold.std()*np.sqrt(252)


std_silver=r_silver.std()*np.sqrt(252)
skew_gold=r_gold.skew()
skew_silver=r_silver.skew()
kurt_gold=r_gold.kurtosis()
kurt_silver=r_silver.kurtosis()


def distance_corr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    n = X.shape[0]
    a = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
    b = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(2))

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov = np.sqrt((A * B).mean())
    dvar_x = np.sqrt((A * A).mean())
    dvar_y = np.sqrt((B * B).mean())

    return dcov / np.sqrt(dvar_x * dvar_y)

pearson_rho_prix=pearsonr(gold.values[0:5026].ravel(), silver.values[0:5025].ravel())[0]
spearman_rho_prix=spearmanr(gold.values[0:5026].ravel(), silver.values[0:5025].ravel())[0]
dist_cor_prix=distance_corr(gold.values[0:5026].ravel(), silver.values[0:5025].ravel())
print(pearson_rho_prix, spearman_rho_prix, dist_cor_prix)
pearson_rho_rdts=pearsonr(r_gold[0:5023], r_silver[0:5024])[0]
spearman_rho_rdts=spearmanr(r_gold[0:5023], r_silver[0:5024])[0]
dist_cor_rdts=distance_corr(r_gold[0:5023], r_silver[0:5024])
print(pearson_rho_rdts, spearman_rho_rdts, dist_cor_rdts)
