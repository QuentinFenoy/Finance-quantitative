import numpy as np
import pandas as pd
import scipy.stats as stats

# ---------- Utilities ----------
import numpy as np
import pandas as pd

import yfinance as yf


#matplotlib.use("Qt5Agg")  # backend interactif
import matplotlib.pyplot as plt






# 1) Récupérer données (exemple daily, 10 ans)
tickers = {'gold':'GC=F', 'silver':'SI=F'}
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

# ---------- Helper: ensure returns series ----------
def ensure_series(returns):
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()
    return returns.dropna()

# ---------- Historical VaR & ES (losses are positive) ----------
def var_historical(returns, alpha=0.95):
    returns = ensure_series(returns)
    losses = -returns
    q = losses.quantile(alpha)   # CORRECTION: quantile(alpha) of losses
    return float(q)

def es_historical(returns, alpha=0.95):
    returns = ensure_series(returns)
    losses = -returns
    q = losses.quantile(alpha)
    tail = losses[losses >= q]
    if len(tail)==0:
        return float(q)
    return float(tail.mean())

# ---------- Parametric Normal VaR & ES (on returns) ----------
def var_normal(returns, alpha=0.95):
    returns = ensure_series(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    # quantile of returns at prob 1-alpha:
    z = stats.norm.ppf(1-alpha)           # e.g. ppf(0.05) for alpha=0.95
    r_quantile = mu + sigma * z           # return quantile at 1-alpha
    var = - r_quantile                     # VaR = -quantile_return
    return float(var)


def es_normal(returns, alpha=0.95):
    returns = ensure_series(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = stats.norm.ppf(1-alpha)         # e.g. ppf(0.05) when alpha=0.95
    pdf_z = stats.norm.pdf(z)
    # CORRECTION: division par (1-alpha), pas alpha
    es_return = mu - sigma * pdf_z / (1 - alpha)
    es_loss = -es_return
    return float(es_loss)


# ---------- Parametric Student-t VaR & ES (via sampling to avoid formula mistakes) ----------
def var_es_t_via_sim(returns, alpha=0.95, df=5, n_samp=200000, random_state=None):
    rng = np.random.default_rng(random_state)
    returns = ensure_series(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    # standardized t has variance df/(df-2) (if df>2). To obtain desired sigma, scale by:
    scale = sigma / np.sqrt(df/(df-2.0)) if df>2 else np.nan
    # draw samples of returns from t with loc=mu and scale=scale
    t_samples = stats.t.rvs(df, size=n_samp, random_state=rng)
    r_samples = mu + scale * t_samples
    loss_samples = -r_samples
    var = np.quantile(loss_samples, alpha)
    es = loss_samples[loss_samples >= var].mean()
    return float(var), float(es)

# ---------- Monte-Carlo GBM VaR & ES (returns-based) ----------
def var_es_gbm_returns(returns, horizon_days=1, n_sims=200000, alpha=0.95, random_state=None):
    """
    Simule la distribution des log-returns sur 'horizon_days' en partant des statistiques observées.
    Retourne VaR & ES en FRACTION (perte fractionnelle).
    """
    rng = np.random.default_rng(random_state)
    returns = ensure_series(returns)
    mu_daily = returns.mean()            # mean of log-returns per day
    sigma_daily = returns.std(ddof=1)    # std of log-returns per day
    # For horizon n days, sum of daily log-returns ~ Normal(mu_daily * n, sigma_daily^2 * n)
    n = horizon_days
    mean_h = mu_daily * n                # CORRECTION: use observed mean directly
    sd_h = sigma_daily * np.sqrt(n)
    eps = rng.standard_normal(n_sims)
    sim_logrets = mean_h-1/2*sd_h**2 + sd_h * eps    # simulated log-return over horizon
    loss_samples = - sim_logrets         # loss as fraction (since log-return ≈ return for small)
    # If you want exact price losses in $, you could do: S0 - S0*np.exp(sim_logrets)
    var = np.quantile(loss_samples, alpha)
    es = loss_samples[loss_samples >= var].mean()
    return float(var), float(es)

# ---------- Monte-Carlo GBM VaR & ES (dollars) ----------
def var_es_gbm_dollars(S0, returns, horizon_days=1, n_sims=200000, alpha=0.95, random_state=None):
    """
    Simule les prix via GBM (log-returns) et renvoie VaR & ES en $ sur horizon_days.
    """
    rng = np.random.default_rng(random_state)
    returns = ensure_series(returns)
    mu_daily = returns.mean()
    sigma_daily = returns.std(ddof=1)
    n = horizon_days
    mean_h = mu_daily * n
    sd_h = sigma_daily * np.sqrt(n)
    eps = rng.standard_normal(n_sims)
    sim_logrets = mean_h-1/2*sd_h**2 + sd_h * eps
    S_end = S0 * np.exp(sim_logrets)
    losses_dollars = S0 - S_end
    var_d = np.quantile(losses_dollars, alpha)
    es_d = losses_dollars[losses_dollars >= var_d].mean()
    # If you also want fractional VaR:
    frac_var = var_d / S0
    frac_es = es_d / S0
    return float(var_d), float(es_d), float(frac_var), float(frac_es)

def compute_exceptions(returns, var_value):
    """
    returns: pd.Series ou array des rendements (log-returns)
    var_value: VaR en terme de perte positive (ex: var_historical renvoyée)
    retourne: exceptions_array (0/1 pd.Series), nb_exceptions (int)
    """
    returns = ensure_series(returns)
    losses = -returns
    exceptions = (losses > var_value).astype(int)
    return exceptions, int(exceptions.sum())

# ---------- Kupiec (LR_uc) robuste ----------
def kupiec_LR_from_array(exceptions_array, p):
    """
    exceptions_array: array-like 0/1
    p: probabilité attendue d'exception (1 - alpha)
    retourne: LR_uc (float), p_value (float), x, n
    """
    exc = np.asarray(exceptions_array).astype(int)
    n = exc.size
    if n == 0:
        return None, None, 0, 0
    x = int(exc.sum())

    # log-vraisemblance sous H0 (p) :
    # logL0 = (n-x) * log(1-p) + x * log(p)
    # logL1 = (n-x) * log(1-pi_hat) + x * log(pi_hat)
    logL0 = (n - x) * math.log(max(1 - p, 1e-300)) + x * math.log(max(p, 1e-300))

    # cas limites
    if x == 0 or x == n:
        # quand pi_hat == 0 ou 1, logL1 = 0 (car L1 = 1)
        logL1 = 0.0
    else:
        pi_hat = x / n
        logL1 = (n - x) * math.log(1 - pi_hat) + x * math.log(pi_hat)

    LR_uc = -2.0 * (logL0 - logL1)
    # LR_uc peut être très grand => p-val proche de 0
    p_value = 1.0 - stats.chi2.cdf(LR_uc, df=1)
    return float(LR_uc), float(p_value), x, n

# ---------- Christoffersen (LR_ind + LR_cc) robuste ----------
def christoffersen_LR(exceptions_array, p):
    """
    exceptions_array: array-like 0/1
    p: probabilité attendue d'exception (1 - alpha)
    retourne: LR_cc, p_value_cc, LR_uc, LR_ind, transition_counts (dict)
    """
    exc = np.asarray(exceptions_array).astype(int)
    n = exc.size
    if n <= 1:
        return None, None, None, None, {'n00':0,'n01':0,'n10':0,'n11':0}

    # compte des transitions
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        a, b = int(exc[i-1]), int(exc[i])
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        elif a == 1 and b == 1:
            n11 += 1

    # Estimations conditionnelles pi0 = P(1 | previous=0), pi1 = P(1 | previous=1)
    # On calcule les log-vraisemblances en sommant les termes seulement s'ils existent (évite 0*log0)
    def safe_term(count, prob):
        if count == 0:
            return 0.0
        # if prob is 0 -> log -> -inf -> whole logL becomes -inf (handled below)
        if prob <= 0.0:
            return -math.inf
        return count * math.log(prob)

    # Estimations pi0, pi1 (avec précaution si dénominateur 0)
    denom0 = n00 + n01
    denom1 = n10 + n11
    pi0 = n01 / denom0 if denom0 > 0 else 0.0
    pi1 = n11 / denom1 if denom1 > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    # logL_ind (modèle Markov avec pi0, pi1)
    logL_ind = 0.0
    # terms: n00 * log(1-pi0) + n01 * log(pi0) + n10 * log(1-pi1) + n11 * log(pi1)
    t1 = safe_term(n00, max(1 - pi0, 0.0))
    t2 = safe_term(n01, max(pi0, 0.0))
    t3 = safe_term(n10, max(1 - pi1, 0.0))
    t4 = safe_term(n11, max(pi1, 0.0))
    logL_ind = t1 + t2 + t3 + t4

    # logL_uncond (modèle inconditionnel avec prob pi)
    logL_uncond = 0.0
    t5 = safe_term(n00 + n10, max(1 - pi, 0.0))
    t6 = safe_term(n01 + n11, max(pi, 0.0))
    logL_uncond = t5 + t6

    # Si une des log-vraisemblances est -inf (probabilité nulle pour un évènement observé)
    if math.isinf(logL_ind) and logL_ind < 0:
        LR_ind = math.inf
    else:
        LR_ind = -2.0 * (logL_uncond - logL_ind)

    # Kupiec LR_uc (basé sur la fréquence globale)
    LR_uc, pval_uc, x_total, n_total = kupiec_LR_from_array(exc, p)

    # combiner
    if math.isinf(LR_ind) or math.isinf(LR_uc):
        LR_cc = math.inf
        pval_cc = 0.0
    else:
        LR_cc = LR_uc + LR_ind
        pval_cc = 1.0 - stats.chi2.cdf(LR_cc, df=2)

    transitions = {'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11}
    return float(LR_cc), float(pval_cc), float(LR_uc), float(LR_ind), transitions





# r_gold : Series de log-returns journaliers
S0 = 4000  # prix spot actuel
alpha = 0.95

# supposons r_gold (Series) et S0 (float) déjà définis


vh = var_historical(r_r_gold, alpha=alpha)
eh = es_historical(r_r_gold, alpha=alpha)

vn = var_normal(r_r_gold, alpha=alpha)
en = es_normal(r_r_gold, alpha=alpha)

vt, et = var_es_t_via_sim(r_r_gold, alpha=alpha, df=5, n_samp=100000, random_state=42)

vm_frac, em_frac = var_es_gbm_returns(r_r_gold, horizon_days=1, n_sims=200000, alpha=alpha, random_state=42)
vd_dollars, ed_dollars, vd_frac, ed_frac = var_es_gbm_dollars(S0=1900.0, returns=r_r_gold, horizon_days=1, n_sims=200000, alpha=alpha, random_state=42)

print("Historical VaR (frac):", vh, "ES (frac):", eh)
print("Normal VaR (frac):", vn, "ES (frac):", en)
print("t VaR (frac):", vt, "ES (frac):", et)
print("MC GBM (frac):", vm_frac, em_frac)
print("MC GBM ($):", vd_dollars, ed_dollars, " -> (frac):", vd_frac, ed_frac)
# Application of backtests:
alpha = 0.99
p = 1.0 - alpha

# VaR historique
vh = var_historical(r_r_gold, alpha=alpha)
exc_hist, n_exc_hist = compute_exceptions(r_r_gold, vh)

print("VH:", vh, "Nb exceptions:", n_exc_hist)

LR_uc, pv_uc, x, n = kupiec_LR_from_array(exc_hist, p)
print("--- Kupiec Test (Historical VaR) ---")
print("LR_uc =", LR_uc, "p-value =", pv_uc, "x/n =", x, "/", n)

LR_cc, pv_cc, LR_uc_val, LR_ind, transitions = christoffersen_LR(exc_hist, p)
print("\n--- Christoffersen Test (Historical VaR) ---")
print("LR_cc =", LR_cc, "p-value =", pv_cc) #if p-value>0.05 : model OK (if LR< approx 6)
print("LR_uc =", LR_uc_val, "LR_ind =", LR_ind)
print("transitions:", transitions)


#EWMA:
from scipy.stats import norm

# ------------------------------------------------------------
# 1. Paramètres
# ------------------------------------------------------------




def portfolio_returns(returns_df, weights):

    # Vérification des dimensions
    if len(weights) != returns_df.shape[1]:
        raise ValueError("Le nombre de poids doit correspondre au nombre d'actifs (colonnes) dans returns_df")
    
    # Calcul du rendement pondéré
    port_returns = returns_df.dot(weights)
    
    # Retour en pd.Series avec index des dates
    port_returns = pd.Series(port_returns, index=returns_df.index, name="Portfolio_Return")
    
    return port_returns



def compute_exceptions(portfolio_returns, VaR_array):
    """
    Compte les exceptions pour un portefeuille.
    
    Parameters
    ----------
    portfolio_returns : pd.Series ou pd.DataFrame
        Rendements journaliers du portefeuille. Si DataFrame multi-actifs, 
        il faut que ce soit déjà le rendement agrégé du portefeuille.
    VaR_array : np.array ou pd.Series
        VaR journalière du portefeuille, même taille que portfolio_returns.
    
    Returns
    -------
    exceptions : np.array
        1 si r < -VaR, 0 sinon
    n_exceptions : int
        Nombre total d'exceptions
    """
    
    # S'assurer que les tailles correspondent
    if len(portfolio_returns) != len(VaR_array):
        raise ValueError("portfolio_returns et VaR_array doivent avoir la même longueur")
    
    # Calcul des exceptions
    exceptions = (portfolio_returns < -VaR_array).astype(int)
    n_exceptions = np.sum(exceptions)
    
    return exceptions, n_exceptions
    
tickers = ["GC=F", "SI=F", "^GSPC"]   # Or, Argent, S&P500 (yfinance)
weights = np.array([0.4, 0.3, 0.3])   # Poids du portefeuille (modifiable)
lambda_ = 0.94                        # Paramètre EWMA RiskMetrics
alpha = 0.95                # Quantile pour la VaR (modifiable)


# ------------------------------------------------------------
# 2. Télécharger les prix
# ------------------------------------------------------------

data = yf.download(tickers, start="2005-12-05")["Close"]
returns = np.log(data / data.shift(1)).dropna()
pf_returns=portfolio_returns(returns, weights)


# ------------------------------------------------------------
# 3. Calcul EWMA de RiskMetrics
# ------------------------------------------------------------


def ewma_cov_matrix(returns, lambda_):
    T, N = returns.shape
    cov_matrix = [np.cov(returns.values, rowvar=False)]  # initialisation avec covariance empirique historique

    for t in range(1, T):
        r_tm1 = returns.iloc[t-1].values.reshape(-1, 1)  # rendement t-1
        cov_matrix.append(lambda_ * cov_matrix[t-1] + (1 - lambda_) * (r_tm1 @ r_tm1.T)) #Formule: \Sigma_t=\lambda\Sigma_{t-1}+(1-\lambda)r_{t-1}r_{t-1}^T

    return cov_matrix


cov_ewma = ewma_cov_matrix(returns, lambda_)

# ------------------------------------------------------------
# 4. Volatilité du portefeuille
# ------------------------------------------------------------

portfolio_variance = np.zeros(len(cov_ewma))
portfolio_vol = np.zeros(len(cov_ewma))
VaR_array = np.zeros(len(cov_ewma))
VaR_pct_loss = np.zeros(len(cov_ewma))
for i in range(len(portfolio_variance)):
    portfolio_variance[i]=weights @ cov_ewma[i] @ weights.T
    portfolio_vol[i]=np.sqrt(portfolio_variance[i])

    # ------------------------------------------------------------
    # 5. Value-at-Risk (VaR)
    # ------------------------------------------------------------

    VaR_array[i] = norm.ppf(alpha) * portfolio_vol[i]  # VaR en log-return

    # Convertir en pourcentage de perte (approximation)
    VaR_pct_loss[i] = -(np.exp(-VaR_array[i]) - 1)

# ------------------------------------------------------------
# 6. Affichage des résultats
# ------------------------------------------------------------

print("Matrice de covariance EWMA :")
print(pd.DataFrame(cov_ewma[-1], index=tickers, columns=tickers))

print("\nVolatilité journalière du portefeuille :", round(portfolio_vol[-1], 6))
print(f"VaR journalière (quantile {alpha}): {VaR_array[-1]:.6f}")
print(f"VaR journalière en % de perte approx : {VaR_pct_loss[-1]*100:.3f}%")
#Backtest EWMA


p = 1.0 - alpha

exc_hist, n_exc_hist = compute_exceptions(pf_returns, VaR_array)

print("VH:", VaR_array, "Nb exceptions:", n_exc_hist)

LR_uc, pv_uc, x, n = kupiec_LR_from_array(exc_hist, p)
print("--- Kupiec Test (EWMA VaR) ---")
print("LR_uc =", LR_uc, "p-value =", pv_uc, "x/n =", x, "/", n)

LR_cc, pv_cc, LR_uc_val, LR_ind, transitions = christoffersen_LR(exc_hist, p)
print("\n--- Christoffersen Test (EWMA VaR) ---")
print("LR_cc =", LR_cc, "p-value =", pv_cc)
print("LR_uc =", LR_uc_val, "LR_ind =", LR_ind)
print("transitions:", transitions)
