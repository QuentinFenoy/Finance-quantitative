"""
Optimiseur de Portefeuille — Mean-Variance (Black-Litterman) + Hierarchical
Risk Parity + Risk Parity, avec backtest walk-forward réaliste (coûts de
transaction inclus) et comparaison multi-stratégies.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.linalg import solve as lin_solve
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. TÉLÉCHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

_RETURNS_CACHE:  dict[tuple, pd.DataFrame]  = {}
_FX_CACHE:       dict[tuple, pd.Series]     = {}
_CCY_CACHE:      dict[str, str | None]      = {}
_MKTCAP_CACHE:   dict[str, float | None]    = {}
_ANALYST_CACHE:  dict[str, float | None]    = {}
_PRICE_CACHE:    dict[str, float | None]    = {}


def _extract_close(raw: pd.DataFrame, ticker: str | None = None) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        return close[ticker] if ticker is not None and ticker in close.columns else close.iloc[:, 0]
    return raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]


def _get_ticker_currency(ticker: str) -> str | None:
    if ticker in _CCY_CACHE:
        return _CCY_CACHE[ticker]
    ccy = None
    try:
        ccy = yf.Ticker(ticker).fast_info.get("currency")
    except Exception:
        pass
    _CCY_CACHE[ticker] = ccy
    return ccy


def _fetch_fx_series(target_currency: str, source_currency: str, period: str) -> pd.Series | None:
    fx_ticker = f"{target_currency}{source_currency}=X"
    key = (fx_ticker, period)
    if key in _FX_CACHE:
        return _FX_CACHE[key]
    try:
        fx_raw = yf.download(fx_ticker, period=period, auto_adjust=True, progress=False)
        if fx_raw.empty:
            raise ValueError("série vide")
        fx_close = _extract_close(fx_raw, fx_ticker)
    except Exception as e:
        print(f"   ⚠️  Impossible de récupérer {fx_ticker} ({e})")
        return None
    _FX_CACHE[key] = fx_close
    return fx_close


def _convert_prices_to_currency(prices: pd.DataFrame, tickers: list[str],
                                  target_currency: str, period: str) -> pd.DataFrame:
    converted = prices.copy()
    for ticker in tickers:
        if ticker not in converted.columns:
            continue
        ccy = _get_ticker_currency(ticker)
        if ccy is None or ccy == target_currency:
            continue
        fx_close = _fetch_fx_series(target_currency, ccy, period)
        if fx_close is None:
            continue
        fx_aligned = fx_close.reindex(converted.index).ffill().bfill()
        converted[ticker] = converted[ticker] / fx_aligned
        print(f"   💱 {ticker} : {ccy} → {target_currency} (taux {target_currency}{ccy}=X)")
    return converted

def _convert_market_cap_to_currency(
    market_cap: float,
    from_currency: str | None,
    target_currency: str,
    period: str = "1mo",
) -> float:
    """
    Convertit une capitalisation boursière vers la devise cible.
    """

    if (
        market_cap is None
        or from_currency is None
        or from_currency == target_currency
    ):
        return market_cap

    fx_close = _fetch_fx_series(target_currency, from_currency, period)

    if fx_close is None or fx_close.empty:
        return market_cap

    fx = float(fx_close.iloc[-1])

    return market_cap / fx

def fetch_returns(
    tickers: list[str], period: str = "5y", use_cache: bool = True,
    target_currency: str = "EUR",
) -> pd.DataFrame:
    key = (tuple(tickers), period, target_currency)
    if use_cache and key in _RETURNS_CACHE:
        return _RETURNS_CACHE[key].copy()

    print(f"\n📥 Téléchargement des données ({period}) pour : {', '.join(tickers)}")
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if len(tickers) == 1 else raw

    prices.dropna(how="all", inplace=True)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Données manquantes pour : {missing}")

    print(f"   🌍 Conversion des prix en {target_currency}...")
    prices = _convert_prices_to_currency(prices, tickers, target_currency, period)
    returns = np.log(prices / prices.shift(1)).dropna()

    print(f"   ✔ {len(returns)} jours de trading chargés "
          f"({returns.index[0].date()} → {returns.index[-1].date()})")

    _RETURNS_CACHE[key] = returns.copy()
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# 2. MATRICE DE COVARIANCE — LEDOIT-WOLF SHRINKAGE
# ─────────────────────────────────────────────────────────────────────────────

def covariance_ledoit_wolf(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    lw = LedoitWolf()
    lw.fit(returns)
    return lw.covariance_ * 252, lw.shrinkage_


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRIOR DE MARCHÉ — CAPITALISATIONS BOURSIÈRES RÉELLES
# ─────────────────────────────────────────────────────────────────────────────

def fetch_market_cap_weights(
    tickers: list[str],
    fallback_weight: float | None = None,
) -> np.ndarray:
    """
    Double tentative : fast_info d'abord, puis info["marketCap"] en fallback.
    """
    caps, missing_idx = [], []

    for i, ticker in enumerate(tickers):
        if ticker not in _MKTCAP_CACHE:
            mc = None
            try:
                mc = yf.Ticker(ticker).fast_info.get("market_cap")
                if not mc or mc <= 0:
                    mc = yf.Ticker(ticker).info.get("marketCap")
            except Exception:
                pass
            _MKTCAP_CACHE[ticker] = mc if (mc and mc > 0) else None

        mc = _MKTCAP_CACHE[ticker]

        if mc:
            currency = _get_ticker_currency(ticker)

            mc = _convert_market_cap_to_currency(
                market_cap=float(mc),
                from_currency=currency,
                target_currency="EUR",
            )

            caps.append(mc)
        else:
            caps.append(None)
            missing_idx.append(i)

    caps_arr  = np.array([c if c is not None else 0.0 for c in caps], dtype=float)
    valid_mask = caps_arr > 0

    if valid_mask.sum() == 0:
        print("   ⚠️  Aucune market cap disponible → poids équipondérés.")
        return np.ones(len(tickers)) / len(tickers)

    if missing_idx:
        missing_tickers = [tickers[i] for i in missing_idx]
        print(f"   ℹ️  Pas de market cap pour : {', '.join(missing_tickers)} → fallback médiane.")

        if fallback_weight is not None:
            fill = (fallback_weight * caps_arr[valid_mask].sum()
                    / max(1 - fallback_weight * len(missing_idx), 1e-9))
        else:
            # CORRECTION : médiane au lieu de moyenne
            fill = float(np.median(caps_arr[valid_mask]))

        for i in missing_idx:
            caps_arr[i] = fill

    return caps_arr / caps_arr.sum()


# ─────────────────────────────────────────────────────────────────────────────
# 4. SOURCES DE VIEWS — MOMENTUM, ANALYST, ML
# ─────────────────────────────────────────────────────────────────────────────

def _z_normalize(arr: np.ndarray, clip: float = 2.0) -> np.ndarray:
    std = arr.std()
    if std < 1e-10:
        return np.zeros_like(arr)
    return np.clip((arr - arr.mean()) / std, -clip, clip)


# ── 4a. Momentum 12-1 ────────────────────────────────────────────────────────

def views_from_momentum(
    returns: pd.DataFrame,
    cov_matrix: np.ndarray,
    lookback_long: int = 252,
    skip_recent: int = 21,
    signal_scale: float = 0.07,
    min_signal_threshold: float = 0.02,
    view_confidence_scale: float = 1.5,
    z_clip: float = 2.0,
    q_cap: float = 0.15,
    tau: float = 0.05,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    n = len(returns.columns)
    if len(returns) < lookback_long + skip_recent:
        return None, None, None

    window     = returns.iloc[-(lookback_long):-(skip_recent) if skip_recent > 0 else len(returns)]
    raw_signal = window.sum().values
    z          = _z_normalize(raw_signal, clip=z_clip)
    q_raw      = np.clip(z * signal_scale, -q_cap, q_cap)

    valid = np.abs(z) >= min_signal_threshold
    if valid.sum() == 0:
        return None, None, None

    sel = np.where(valid)[0]
    P   = np.zeros((len(sel), n))
    Q   = np.zeros(len(sel))
    for row, col in enumerate(sel):
        P[row, col] = 1.0
        Q[row]      = q_raw[col]

    Omega = np.diag(np.diag(P @ cov_matrix @ P.T) * view_confidence_scale * tau)
    return P, Q, Omega


# ── 4b. Analyst price targets ────────────────────────────────────────────────

_EQUITY_TICKERS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "SAP.DE", "ASML.AS", "MC.PA", "NESN.SW",
}


def fetch_analyst_implied_returns(
    tickers: list[str],
    returns: pd.DataFrame,
    q_cap: float = 0.20,
    analyst_confidence_scale: float = 2.5,
    horizon_years: float = 1.0,
) -> dict[str, float]:
    result = {}
    for ticker in tickers:
        if ticker not in _EQUITY_TICKERS:
            continue

        if ticker not in _PRICE_CACHE:
            try:
                price = yf.Ticker(ticker).fast_info.get("last_price")
            except Exception:
                price = None
            _PRICE_CACHE[ticker] = price

        if ticker not in _ANALYST_CACHE:
            target = None
            try:
                info   = yf.Ticker(ticker).info
                target = info.get("targetMedianPrice") or info.get("targetMeanPrice")
            except Exception:
                pass
            _ANALYST_CACHE[ticker] = target

        price  = _PRICE_CACHE[ticker]
        target = _ANALYST_CACHE[ticker]

        if price and target and price > 0 and target > 0:
            r_impl         = (target - price) / price / horizon_years
            result[ticker] = float(np.clip(r_impl, -q_cap, q_cap))

    return result


def views_from_analyst(
    tickers: list[str],
    returns: pd.DataFrame,
    cov_matrix: np.ndarray,
    q_cap: float = 0.20,
    analyst_confidence_scale: float = 2.5,
    tau: float = 0.05,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    implied = fetch_analyst_implied_returns(tickers, returns, q_cap=q_cap,
                                            analyst_confidence_scale=analyst_confidence_scale)
    if not implied:
        return None, None, None

    n           = len(tickers)
    ticker_idx  = {t: i for i, t in enumerate(tickers)}
    valid_ticks = [t for t in tickers if t in implied]

    k = len(valid_ticks)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    for row, t in enumerate(valid_ticks):
        P[row, ticker_idx[t]] = 1.0
        Q[row]                = implied[t]

    Omega = np.diag(np.diag(P @ cov_matrix @ P.T) * analyst_confidence_scale * tau)
    return P, Q, Omega


# ── 4c. ML Ridge cross-sectionnel ────────────────────────────────────────────

def _build_ml_features(returns: pd.DataFrame, end_idx: int) -> np.ndarray | None:
    min_required = 252 + 14
    if end_idx < min_required:
        return None

    n          = len(returns.columns)
    n_features = 7
    X          = np.zeros((n, n_features))

    for j, col in enumerate(returns.columns):
        series = returns[col].iloc[:end_idx].values

        # [0] Momentum 12m-1m (Jegadeesh & Titman)
        mom_12m  = series[-252:].sum() if len(series) >= 252 else 0.0
        mom_1m   = series[-21:].sum()  if len(series) >= 21  else 0.0
        X[j, 0]  = mom_12m - mom_1m

        # [1] Volatilité réalisée 1m annualisée
        X[j, 1]  = series[-21:].std() * np.sqrt(252) if len(series) >= 21 else 0.0

        # [2] Volatilité réalisée 3m annualisée
        X[j, 2]  = series[-63:].std() * np.sqrt(252) if len(series) >= 63 else 0.0

        # [3] RSI 14j normalisé [0, 1] (Wilder)
        delta    = series[-15:] if len(series) >= 15 else series
        gains    = np.where(delta > 0, delta, 0.0)
        losses   = np.where(delta < 0, -delta, 0.0)
        rs       = (gains.mean() + 1e-10) / (losses.mean() + 1e-10)
        X[j, 3]  = 1.0 - 1.0 / (1.0 + rs)

        # [4] Ratio vol 1m / vol 3m — stress récent vs normal
        X[j, 4]  = X[j, 1] / X[j, 2] if X[j, 2] > 1e-10 else 1.0

        # [5] Skewness 1m
        r1m      = series[-21:] if len(series) >= 21 else series
        std      = r1m.std()
        X[j, 5]  = float(((r1m - r1m.mean()) ** 3).mean() / (std ** 3 + 1e-10))

        # [6] Mean reversion 1m (DeBondt & Thaler 1985)
        X[j, 6]  = -mom_1m

    # ── Normalisation cross-sectionelle (CORRECTION 19) ──────────────────────
    # Z-score entre actifs, feature par feature, à cette date.
    # Résultat : X[j, f] = "l'actif j est à +X écarts-types de la moyenne
    # cross-sectionelle sur la feature f à cette date".
    for f in range(n_features):
        col_f  = X[:, f]
        std_cs = col_f.std()
        if std_cs > 1e-10:
            X[:, f] = (col_f - col_f.mean()) / std_cs
        # Si std ≈ 0 (tous identiques), on laisse à 0 — pas de signal

    return X


def views_from_ml(
    returns: pd.DataFrame,
    cov_matrix: np.ndarray,
    forward_window: int = 21,
    min_train_samples: int = 20,
    ridge_alpha: float = 1.0,
    signal_scale: float = 0.06,
    z_clip: float = 2.0,
    q_cap: float = 0.12,
    view_confidence_scale: float = 2.0,
    tau: float = 0.05,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    n            = len(returns.columns)
    T            = len(returns)
    min_required = 252 + 14 + forward_window  # cohérent avec _build_ml_features

    if T < min_required + min_train_samples * forward_window:
        return None, None, None

    X_list, y_list = [], []
    step = forward_window

    for end in range(min_required, T - forward_window, step):
        X_t = _build_ml_features(returns, end)
        if X_t is None:
            continue
        y_t = returns.iloc[end:end + forward_window].sum().values
        X_list.append(X_t)
        y_list.append(y_t)

    if len(X_list) < min_train_samples:
        return None, None, None

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    n_train_obs  = len(X_list) - 1
    X_train      = X_all[:n_train_obs * n]
    y_train      = y_all[:n_train_obs * n]
    X_pred       = X_list[-1]

    # StandardScaler temporel (les features sont déjà cross-sectionellement
    # normalisées dans _build_ml_features — le scaler stabilise inter-périodes)
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled  = scaler.transform(X_pred)

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train_scaled, y_train)

    raw_preds = model.predict(X_pred_scaled)
    z         = _z_normalize(raw_preds, clip=z_clip)
    q_raw     = np.clip(z * signal_scale, -q_cap, q_cap)

    P     = np.eye(n)
    Q     = q_raw
    Omega = np.diag(np.diag(P @ cov_matrix @ P.T) * view_confidence_scale * tau)

    return P, Q, Omega


# ── 4d. Fusion pondérée des trois sources ────────────────────────────────────

def combine_views(
    tickers: list[str],
    cov_matrix: np.ndarray,
    P_mom: np.ndarray | None, Q_mom: np.ndarray | None,
    P_ana: np.ndarray | None, Q_ana: np.ndarray | None,
    P_ml:  np.ndarray | None, Q_ml:  np.ndarray | None,
    w_momentum: float = 0.40,
    w_analyst:  float = 0.35,
    w_ml:       float = 0.25,
    view_confidence_scale: float = 1.5,
    tau: float = 0.05,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    n = len(tickers)

    def extract_q_per_asset(P, Q, n):
        q = np.full(n, np.nan)
        if P is None or Q is None:
            return q
        for row in range(P.shape[0]):
            col = np.argmax(np.abs(P[row]))
            if P[row, col] != 0:
                q[col] = Q[row]
        return q

    q_mom = extract_q_per_asset(P_mom, Q_mom, n)
    q_ana = extract_q_per_asset(P_ana, Q_ana, n)
    q_ml  = extract_q_per_asset(P_ml,  Q_ml,  n)

    Q_fused  = np.zeros(n)
    has_view = np.zeros(n, dtype=bool)

    for i in range(n):
        sources, weights = [], []
        if not np.isnan(q_mom[i]): sources.append(q_mom[i]); weights.append(w_momentum)
        if not np.isnan(q_ana[i]): sources.append(q_ana[i]); weights.append(w_analyst)
        if not np.isnan(q_ml[i]):  sources.append(q_ml[i]);  weights.append(w_ml)

        if sources:
            w_arr         = np.array(weights) / np.sum(weights)
            Q_fused[i]    = float(np.dot(w_arr, sources))
            has_view[i]   = True

    if has_view.sum() == 0:
        return None, None, None

    sel   = np.where(has_view)[0]
    k     = len(sel)
    P_out = np.zeros((k, n))
    Q_out = np.zeros(k)
    for row, col in enumerate(sel):
        P_out[row, col] = 1.0
        Q_out[row]      = Q_fused[col]

    Omega_out = np.diag(np.diag(P_out @ cov_matrix @ P_out.T) * view_confidence_scale * tau)
    return P_out, Q_out, Omega_out


def print_combined_views(tickers, P, Q, q_mom, q_ana, q_ml):
    n = len(tickers)
    print("\n  📊 Views fusionnées (momentum | analyst | ML Ridge) :")
    hdr = f"  {'Actif':<12} {'Fusionnée':>11}  {'Momentum':>10}  {'Analyst':>10}  {'ML Ridge':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    q_fused_by_asset = {np.argmax(np.abs(P[row])): Q[row] for row in range(P.shape[0])}

    for i, t in enumerate(tickers):
        qf  = q_fused_by_asset.get(i)
        qm  = q_mom[i] if not np.isnan(q_mom[i]) else None
        qa  = q_ana[i] if not np.isnan(q_ana[i]) else None
        qml = q_ml[i]  if not np.isnan(q_ml[i])  else None

        def fmt(v):
            if v is None: return "    —    "
            return f"{'▲' if v > 0 else '▼'} {v*100:>+6.2f}%"

        fused_str = f"{'▲' if qf > 0 else '▼'} {qf*100:>+6.2f}%" if qf is not None else "    —    "
        print(f"  {t:<12} {fused_str:>11}  {fmt(qm):>10}  {fmt(qa):>10}  {fmt(qml):>10}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. BLACK-LITTERMAN
# ─────────────────────────────────────────────────────────────────────────────

def black_litterman(
    cov_matrix: np.ndarray,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
    delta: float = 2.5,
    tau: float = 0.05,
    P: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    Omega: np.ndarray | None = None,
    view_confidence_scale: float = 1.0,
    return_posterior_cov: bool = False,
    w_market: np.ndarray | None = None,
):
    n = cov_matrix.shape[0]
    if w_market is None:
        w_market = fetch_market_cap_weights(list(returns.columns))

    pi = delta * cov_matrix @ w_market

    if P is None or Q is None:
        return (pi, np.zeros_like(cov_matrix)) if return_posterior_cov else pi

    P = np.atleast_2d(P).astype(float)
    Q = np.atleast_1d(Q).astype(float)

    if Omega is None:
        Omega = np.diag(np.diag(view_confidence_scale * tau * P @ cov_matrix @ P.T))

    tau_sigma     = tau * cov_matrix
    tau_sigma_inv = lin_solve(tau_sigma, np.eye(n), assume_a="pos")
    omega_inv     = (np.diag(1.0 / np.diag(Omega))
                     if np.allclose(Omega, np.diag(np.diag(Omega)))
                     else lin_solve(Omega, np.eye(Omega.shape[0]), assume_a="pos"))

    M_inv = tau_sigma_inv + P.T @ omega_inv @ P
    rhs   = tau_sigma_inv @ pi + P.T @ omega_inv @ Q

    if return_posterior_cov:
        M = lin_solve(M_inv, np.eye(n), assume_a="pos")
        return M @ rhs, M

    return lin_solve(M_inv, rhs, assume_a="pos")


def shrink_mu(mu: np.ndarray, shrink_factor: float = 1.0) -> np.ndarray:
    if not (0.0 <= shrink_factor <= 1.0):
        raise ValueError("shrink_factor doit être entre 0.0 et 1.0")
    return shrink_factor * mu + (1 - shrink_factor) * np.mean(mu)


def print_bl_summary(tickers, mu_hist, mu_bl, w_market):
    print("\n" + "─" * 78)
    print("  BLACK-LITTERMAN — Comparaison rendements espérés")
    print("─" * 78)
    print(f"  {'Actif':<12} {'w_mkt':>8} {'Historique':>12} {'BL posterior':>14} {'Δ vs hist':>10}")
    print("─" * 78)
    for i, t in enumerate(tickers):
        d = mu_bl[i] - mu_hist[i]
        print(f"  {t:<12} {w_market[i]*100:>7.2f}%  {mu_hist[i]*100:>11.2f}% "
              f"{mu_bl[i]*100:>13.2f}% {d*100:>+9.2f}%")
    print("─" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# 6. HIERARCHICAL RISK PARITY
# ─────────────────────────────────────────────────────────────────────────────

def _hrp_quasi_diagonal_order(link: np.ndarray, n_leaves: int) -> list[int]:
    link         = link.astype(int)
    sorted_items = pd.Series([link[-1, 0], link[-1, 1]])
    while sorted_items.max() >= n_leaves:
        sorted_items.index = range(0, sorted_items.shape[0] * 2, 2)
        clusters           = sorted_items[sorted_items >= n_leaves]
        idx                = clusters.index
        rows               = clusters.values - n_leaves
        sorted_items[idx]  = link[rows, 0]
        right_children     = pd.Series(link[rows, 1], index=idx + 1)
        sorted_items       = pd.concat([sorted_items, right_children]).sort_index()
        sorted_items.index = range(sorted_items.shape[0])
    return sorted_items.tolist()


def _hrp_cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    sub_cov = cov[np.ix_(items, items)]
    ivp     = 1.0 / np.diag(sub_cov)
    ivp    /= ivp.sum()
    return float(ivp @ sub_cov @ ivp)


def hierarchical_risk_parity(cov_matrix: np.ndarray) -> np.ndarray:
    n = cov_matrix.shape[0]
    if n == 1:
        return np.array([1.0])
    std  = np.sqrt(np.diag(cov_matrix))
    corr = np.clip(cov_matrix / np.outer(std, std), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    link = linkage(squareform(dist, checks=False), method="single")
    order   = _hrp_quasi_diagonal_order(link, n)
    weights = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        next_clusters = []
        for items in clusters:
            if len(items) <= 1:
                continue
            mid   = len(items) // 2
            left, right = items[:mid], items[mid:]
            var_l = _hrp_cluster_variance(cov_matrix, left)
            var_r = _hrp_cluster_variance(cov_matrix, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            weights[left]  *= alpha
            weights[right] *= (1.0 - alpha)
            next_clusters  += [left, right]
        clusters = next_clusters
    w = weights.sort_index().values
    return w / w.sum()


# ─────────────────────────────────────────────────────────────────────────────
# 7. RISK PARITY / EQUAL RISK CONTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def equal_risk_contribution(
    cov_matrix: np.ndarray, min_weight: float = 0.0, max_weight: float = 1.0
) -> np.ndarray:
    n = cov_matrix.shape[0]

    def obj(w):
        pv = w @ cov_matrix @ w
        rc = w * (cov_matrix @ w) / pv
        return np.sum((rc - 1.0 / n) ** 2)

    result = minimize(obj, np.ones(n) / n, method="SLSQP",
                      bounds=[(min_weight, max_weight)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                      options={"ftol": 1e-14, "maxiter": 2000})
    if not result.success:
        raise RuntimeError(f"Risk parity : {result.message}")
    return result.x / result.x.sum()


# ─────────────────────────────────────────────────────────────────────────────
# 8. OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

def min_achievable_vol(
    cov_matrix: np.ndarray,
    bounds: list[tuple] | None = None,
    budget_constraint: dict | None = None,
) -> float:
    """
    Volatilité minimale atteignable sous les contraintes de poids.
    """
    n = cov_matrix.shape[0]
    if bounds is None:
        bounds = [(0.0, 1.0)] * n
    if budget_constraint is None:
        budget_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        lambda w: w @ cov_matrix @ w,
        np.ones(n) / n, method="SLSQP",
        bounds=bounds,
        constraints=[budget_constraint],
        options={"ftol": 1e-9},
    )
    return float(np.sqrt(res.fun))


def optimize_portfolio(
    returns: pd.DataFrame,
    cov_matrix: np.ndarray,
    target_volatility: float,
    risk_free_rate: float = 0.04,
    objective: str = "max_sharpe",
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    mu_override: np.ndarray | None = None,
    vol_cov_matrix: np.ndarray | None = None,
) -> dict:
    n       = len(returns.columns)
    mu      = mu_override if mu_override is not None else returns.mean().values * 252
    vol_cov = vol_cov_matrix if vol_cov_matrix is not None else cov_matrix

    if min_weight * n > 1.0:
        raise ValueError(f"Impossible : {n} actifs × {min_weight:.1%} > 100 %.")
    if max_weight < 1.0 / n:
        raise ValueError(f"max_weight={max_weight:.1%} trop faible pour {n} actifs.")

    bounds = [(min_weight, max_weight)] * n
    w0     = np.ones(n) / n
    budget = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def portfolio_vol(w):
        return np.sqrt(float(w @ vol_cov @ w))

    if objective == "hrp":
        w_opt = hierarchical_risk_parity(vol_cov)
        if min_weight > 0.0 or max_weight < 1.0:
            w_opt = np.clip(w_opt, min_weight, max_weight)
            w_opt /= w_opt.sum()

    elif objective == "risk_parity":
        w_opt = equal_risk_contribution(vol_cov, min_weight, max_weight)

    elif objective == "min_variance":
        res = minimize(
            lambda w: w @ vol_cov @ w, w0, method="SLSQP",
            bounds=bounds, constraints=[budget],
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if not res.success:
            raise RuntimeError(f"Optimisation : {res.message}")
        w_opt = res.x

    elif objective == "max_sharpe":
        vmin = min_achievable_vol(vol_cov, bounds, budget)
        if target_volatility < vmin:
            raise ValueError(f"Cible {target_volatility:.1%} < vol min {vmin:.1%}.")
        res = minimize(
            lambda w: -(w @ mu - risk_free_rate) / portfolio_vol(w),
            w0, method="SLSQP", bounds=bounds,
            constraints=[budget,
                         {"type": "ineq", "fun": lambda w: target_volatility - portfolio_vol(w)}],
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if not res.success:
            raise RuntimeError(f"Optimisation : {res.message}")
        w_opt = res.x

    elif objective == "target_vol":
        vmin = min_achievable_vol(vol_cov, bounds, budget)
        if target_volatility < vmin:
            raise ValueError(f"Cible {target_volatility:.1%} < vol min {vmin:.1%}.")

        obj_fn = lambda w: -(w @ mu)

        # CORRECTION : contrainte d'égalité en premier
        res = minimize(
            obj_fn, w0, method="SLSQP", bounds=bounds,
            constraints=[budget,
                         {"type": "eq", "fun": lambda w: portfolio_vol(w) - target_volatility}],
            options={"ftol": 1e-9, "maxiter": 1500},
        )
        if not res.success:
            # Fallback inégalité (comportement original)
            res = minimize(
                obj_fn, w0, method="SLSQP", bounds=bounds,
                constraints=[budget,
                             {"type": "ineq", "fun": lambda w: target_volatility - portfolio_vol(w)}],
                options={"ftol": 1e-9, "maxiter": 1000},
            )
            if not res.success:
                raise RuntimeError(f"Optimisation : {res.message}")
        w_opt = res.x

    else:
        raise ValueError(f"objective inconnu : {objective!r}")

    port_vol = portfolio_vol(w_opt)
    port_ret = float(w_opt @ mu)
    sharpe   = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else np.nan
    return {"weights": w_opt, "return": port_ret, "volatility": port_vol,
            "sharpe": sharpe, "mu_used": mu}


# ─────────────────────────────────────────────────────────────────────────────
# 9. AFFICHAGE
# ─────────────────────────────────────────────────────────────────────────────

def print_results(tickers, total_amount, result, shrinkage, returns,
                  mu_bl=None, w_market=None):
    weights       = result["weights"]
    tickers_clean = list(returns.columns)
    mu_assets     = returns.mean() * 252
    vol_assets    = returns.std() * np.sqrt(252)
    mu_display    = mu_bl if mu_bl is not None else mu_assets.values

    print("\n" + "═" * 55)
    print("       RÉSULTATS DE L'OPTIMISATION DE PORTEFEUILLE")
    print("═" * 55)
    print(f"\n  Shrinkage Ledoit-Wolf : {shrinkage:.4f}")
    print(f"  Rendements espérés    : "
          f"{'Black-Litterman (momentum+analyst+ML)' if mu_bl is not None else 'Moyenne historique'}")
    if w_market is not None:
        print(f"  Prior de marché       : Capitalisations boursières réelles (fallback médiane)")
    print(f"  Rendement espéré      : {result['return']*100:.2f}%")
    print(f"  Volatilité            : {result['volatility']*100:.2f}%")
    print(f"  Sharpe                : {result['sharpe']:.4f}")

    print(f"\n{'─'*80}")
    print(f"  {'Actif':<10} {'Poids':>8}  {'E[r] hist':>10}  {'E[r] BL':>9}  {'Vol an.':>8}  {'Montant':>14}")
    print("─" * 80)
    for i, (ticker, w) in enumerate(zip(tickers_clean, weights)):
        print(f"  {ticker:<10} {w*100:>7.2f}%  {mu_assets[ticker]*100:>9.2f}%  "
              f"{mu_display[i]*100:>8.2f}%  {vol_assets[ticker]*100:>7.2f}%  "
              f"{w*total_amount:>13,.2f} €")
    print("─" * 80)
    total = sum(w * total_amount for w in weights)
    print(f"  {'TOTAL':<10} {'100.00%':>8}  {'':>10}  {'':>9}  {'':>8}  {total:>13,.2f} €")
    print("═" * 80 + "\n")
    print("  Matrice de corrélation (rendements journaliers) :")
    print(returns.corr().round(3).to_string())
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 10. CONSTRUCTION DES VIEWS FUSIONNÉES
# ─────────────────────────────────────────────────────────────────────────────

def _build_combined_views(
    returns: pd.DataFrame,
    cov_matrix: np.ndarray,
    tickers: list[str],
    bl_views_P, bl_views_Q, bl_views_Omega,
    use_momentum_views: bool,
    use_analyst_views: bool,
    use_ml_views: bool,
    momentum_lookback: int, momentum_skip: int,
    momentum_signal_scale: float, momentum_confidence_scale: float,
    z_clip: float, q_cap_momentum: float,
    analyst_q_cap: float, analyst_confidence_scale: float,
    ml_forward_window: int, ml_ridge_alpha: float,
    ml_signal_scale: float, ml_q_cap: float, ml_confidence_scale: float,
    w_momentum: float, w_analyst: float, w_ml: float,
    view_confidence_scale_fused: float,
    tau: float = 0.05,
    verbose: bool = False,
) -> tuple:
    if bl_views_P is not None and bl_views_Q is not None:
        return bl_views_P, bl_views_Q, bl_views_Omega, "utilisateur"

    P_mom, Q_mom, Omega_mom = None, None, None
    P_ana, Q_ana, Omega_ana = None, None, None
    P_ml,  Q_ml,  Omega_ml  = None, None, None

    n         = len(tickers)
    q_mom_arr = np.full(n, np.nan)
    q_ana_arr = np.full(n, np.nan)
    q_ml_arr  = np.full(n, np.nan)

    if use_momentum_views:
        P_mom, Q_mom, Omega_mom = views_from_momentum(
            returns=returns, cov_matrix=cov_matrix,
            lookback_long=momentum_lookback, skip_recent=momentum_skip,
            signal_scale=momentum_signal_scale,
            view_confidence_scale=momentum_confidence_scale,
            z_clip=z_clip, q_cap=q_cap_momentum, tau=tau,
        )
        if P_mom is not None:
            for row in range(P_mom.shape[0]):
                col = np.argmax(np.abs(P_mom[row]))
                q_mom_arr[col] = Q_mom[row]
              
    if use_analyst_views and w_analyst > 0.0:
        P_ana, Q_ana, Omega_ana = views_from_analyst(
            tickers=tickers, returns=returns, cov_matrix=cov_matrix,
            q_cap=analyst_q_cap, analyst_confidence_scale=analyst_confidence_scale,
            tau=tau,
        )
        if P_ana is not None:
            for row in range(P_ana.shape[0]):
                col = np.argmax(np.abs(P_ana[row]))
                q_ana_arr[col] = Q_ana[row]

    if use_ml_views:
        P_ml, Q_ml, Omega_ml = views_from_ml(
            returns=returns, cov_matrix=cov_matrix,
            forward_window=ml_forward_window, ridge_alpha=ml_ridge_alpha,
            signal_scale=ml_signal_scale, q_cap=ml_q_cap,
            view_confidence_scale=ml_confidence_scale, tau=tau,
        )
        if P_ml is not None:
            for row in range(P_ml.shape[0]):
                col = np.argmax(np.abs(P_ml[row]))
                q_ml_arr[col] = Q_ml[row]

    if P_mom is None and P_ana is None and P_ml is None:
        return None, None, None, "aucune (données insuffisantes)"

    P_eff, Q_eff, Omega_eff = combine_views(
        tickers=tickers, cov_matrix=cov_matrix,
        P_mom=P_mom, Q_mom=Q_mom, P_ana=P_ana, Q_ana=Q_ana,
        P_ml=P_ml,   Q_ml=Q_ml,
        w_momentum=w_momentum, w_analyst=w_analyst, w_ml=w_ml,
        view_confidence_scale=view_confidence_scale_fused, tau=tau,
    )

    sources_active = []
    if P_mom is not None and use_momentum_views:              sources_active.append("momentum")
    if P_ana is not None and use_analyst_views and w_analyst > 0: sources_active.append("analyst")
    if P_ml  is not None and use_ml_views:                   sources_active.append("ML Ridge")
    label = "fusionnées (" + " + ".join(sources_active) + ")"

    if verbose and P_eff is not None:
        print_combined_views(tickers, P_eff, Q_eff, q_mom_arr, q_ana_arr, q_ml_arr)

    return P_eff, Q_eff, Omega_eff, label


# ─────────────────────────────────────────────────────────────────────────────
# 11. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_optimizer(
    tickers: list[str], total_amount: float, target_volatility: float,
    period: str = "5y", risk_free_rate: float = 0.04, objective: str = "max_sharpe",
    max_weight: float = 1.0, min_weight: float = 0.0,
    bl_delta: float = 2.5, bl_tau: float = 0.05,
    bl_views_P: np.ndarray | None = None,
    bl_views_Q: np.ndarray | None = None,
    bl_views_Omega: np.ndarray | None = None,
    bl_view_confidence_scale: float = 1.0,
    mu_shrink_factor: float = 1.0,
    use_posterior_cov: bool = False,
    use_momentum_views: bool = True,
    momentum_lookback: int = 252, momentum_skip: int = 21,
    momentum_signal_scale: float = 0.07, momentum_confidence_scale: float = 1.5,
    z_clip: float = 2.0, q_cap_momentum: float = 0.15,
    use_analyst_views: bool = True,
    analyst_q_cap: float = 0.20, analyst_confidence_scale: float = 2.5,
    use_ml_views: bool = True,
    ml_forward_window: int = 21, ml_ridge_alpha: float = 1.0,
    ml_signal_scale: float = 0.25, ml_q_cap: float = 0.12,
    ml_confidence_scale: float = 2.0,
    w_momentum: float = 0.40, w_analyst: float = 0.35, w_ml: float = 0.25,
    view_confidence_scale_fused: float = 1.5,
):
    returns = fetch_returns(tickers, period)
    # [CORRECTION 21] Réalignement : yfinance trie les colonnes alphabétiquement.
    # `tickers` doit refléter cet ordre pour tout calcul en aval (market caps,
    # views, BL prior). Sans ça, fetch_market_cap_weights(tickers) renvoie un
    # vecteur dans l'ordre utilisateur, désaligné de cov_matrix.
    tickers = list(returns.columns)

    cov_matrix, shrink = covariance_ledoit_wolf(returns)
    w_market   = fetch_market_cap_weights(tickers)

    P_eff, Q_eff, Omega_eff, views_label = _build_combined_views(
        returns=returns, cov_matrix=cov_matrix, tickers=tickers,
        bl_views_P=bl_views_P, bl_views_Q=bl_views_Q, bl_views_Omega=bl_views_Omega,
        use_momentum_views=use_momentum_views, use_analyst_views=use_analyst_views,
        use_ml_views=use_ml_views,
        momentum_lookback=momentum_lookback, momentum_skip=momentum_skip,
        momentum_signal_scale=momentum_signal_scale,
        momentum_confidence_scale=momentum_confidence_scale,
        z_clip=z_clip, q_cap_momentum=q_cap_momentum,
        analyst_q_cap=analyst_q_cap, analyst_confidence_scale=analyst_confidence_scale,
        ml_forward_window=ml_forward_window, ml_ridge_alpha=ml_ridge_alpha,
        ml_signal_scale=ml_signal_scale, ml_q_cap=ml_q_cap,
        ml_confidence_scale=ml_confidence_scale,
        w_momentum=w_momentum, w_analyst=w_analyst, w_ml=w_ml,
        view_confidence_scale_fused=view_confidence_scale_fused, tau=bl_tau,
        verbose=True,
    )
    print(f"\n  ℹ️  Source des views BL : {views_label}")

    mu_hist = returns.mean().values * 252

    # Black-Litterman retourne des rendements excédentaires
    mu_bl_excess, M_post = black_litterman(
        cov_matrix=cov_matrix,
        returns=returns,
        risk_free_rate=risk_free_rate,
        delta=bl_delta,
        tau=bl_tau,
        P=P_eff,
        Q=Q_eff,
        Omega=Omega_eff,
        view_confidence_scale=bl_view_confidence_scale,
        return_posterior_cov=True,
        w_market=w_market,
    )

    # Conversion en rendements absolus
    mu_bl = mu_bl_excess + risk_free_rate

    print_bl_summary(tickers, mu_hist, mu_bl, w_market)

    mu_for_opt = shrink_mu(mu_bl, mu_shrink_factor)
    if mu_shrink_factor < 1.0:
        print(f"  ℹ️  mu_shrink_factor={mu_shrink_factor}")

    vol_cov = (cov_matrix + M_post) if use_posterior_cov else None

    label = {"min_variance": "Variance minimale", "risk_parity": "Risk Parity (ERC)",
             "hrp": "Hierarchical Risk Parity"}.get(objective)
    if label:
        print(f"\n🎯 Objectif : {label}  |  Rf : {risk_free_rate*100:.1f}%")
    else:
        lbl = "Sharpe max" if objective == "max_sharpe" else "Rendement max @ vol cible"
        print(f"\n🎯 Objectif : {lbl}  |  Vol cible : {target_volatility*100:.1f}%"
              f"  |  Rf : {risk_free_rate*100:.1f}%")

    result = optimize_portfolio(
        returns=returns, cov_matrix=cov_matrix, target_volatility=target_volatility,
        risk_free_rate=risk_free_rate, objective=objective, max_weight=max_weight,
        min_weight=min_weight, mu_override=mu_for_opt, vol_cov_matrix=vol_cov,
    )
    print_results(tickers, total_amount, result, shrink, returns,
                  mu_bl=mu_bl, w_market=w_market)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 12. BACKTEST WALK-FORWARD
# ─────────────────────────────────────────────────────────────────────────────

def perf_metrics(daily_log_rets: pd.Series, risk_free_rate: float = 0.04) -> dict:
    ann_vol        = daily_log_rets.std() * np.sqrt(252)
    ann_ret_log    = daily_log_rets.mean() * 252
    ann_ret_simple = ann_ret_log + 0.5 * ann_vol ** 2  # correction d'Itô

    sharpe  = (ann_ret_simple - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    equity  = np.exp(daily_log_rets.cumsum())
    n_years = len(daily_log_rets) / 252
    cagr    = float(equity.iloc[-1] ** (1.0 / n_years) - 1) if n_years > 0 else np.nan
    max_dd  = (equity / equity.cummax() - 1).min()

    return {"rendement_log": ann_ret_log, "rendement_simple": ann_ret_simple,
            "cagr": cagr, "volatilite": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd}


def backtest_oos(
    returns: pd.DataFrame, target_volatility: float, risk_free_rate: float = 0.04,
    objective: str = "target_vol", lookback: int = 504, rebalance: int = 63,
    max_weight: float = 1.0, min_weight: float = 0.0,
    bl_delta: float = 2.5, bl_tau: float = 0.05,
    bl_views_P=None, bl_views_Q=None, bl_views_Omega=None,
    bl_view_confidence_scale: float = 1.0, mu_shrink_factor: float = 1.0,
    use_posterior_cov: bool = False, transaction_cost_bps: float = 0.0,
    use_momentum_views: bool = True,
    momentum_lookback: int = 252, momentum_skip: int = 21,
    momentum_signal_scale: float = 0.07, momentum_confidence_scale: float = 1.5,
    z_clip: float = 2.0, q_cap_momentum: float = 0.15,
    use_analyst_views: bool = True,
    analyst_q_cap: float = 0.20, analyst_confidence_scale: float = 2.5,
    use_ml_views: bool = True,
    ml_forward_window: int = 21, ml_ridge_alpha: float = 1.0,
    ml_signal_scale: float = 0.25, ml_q_cap: float = 0.12,
    ml_confidence_scale: float = 2.0,
    w_momentum: float = 0.40, w_analyst: float = 0.35, w_ml: float = 0.25,
    view_confidence_scale_fused: float = 1.5,
    w_market: np.ndarray | None = None,
) -> tuple[pd.DataFrame, list]:

    n       = returns.shape[1]
    tickers = list(returns.columns)
    if w_market is None:
        w_market = fetch_market_cap_weights(tickers)

    strat, eqw, idx, w_hist = [], [], [], []
    # Évite un coût fictif de ~100% × cost_bps au premier rebalancement.
    prev_weights = np.ones(n) / n

    start = lookback
    while start < len(returns):
        train = returns.iloc[start - lookback:start]
        test  = returns.iloc[start:start + rebalance]
        if len(test) == 0:
            break

        cov, _ = covariance_ledoit_wolf(train)

        P_eff, Q_eff, Omega_eff, _ = _build_combined_views(
            returns=train, cov_matrix=cov, tickers=tickers,
            bl_views_P=bl_views_P, bl_views_Q=bl_views_Q, bl_views_Omega=bl_views_Omega,
            use_momentum_views=use_momentum_views, use_analyst_views=use_analyst_views,
            use_ml_views=use_ml_views,
            momentum_lookback=momentum_lookback, momentum_skip=momentum_skip,
            momentum_signal_scale=momentum_signal_scale,
            momentum_confidence_scale=momentum_confidence_scale,
            z_clip=z_clip, q_cap_momentum=q_cap_momentum,
            analyst_q_cap=analyst_q_cap, analyst_confidence_scale=analyst_confidence_scale,
            ml_forward_window=ml_forward_window, ml_ridge_alpha=ml_ridge_alpha,
            ml_signal_scale=ml_signal_scale, ml_q_cap=ml_q_cap,
            ml_confidence_scale=ml_confidence_scale,
            w_momentum=w_momentum, w_analyst=w_analyst, w_ml=w_ml,
            view_confidence_scale_fused=view_confidence_scale_fused, tau=bl_tau,
            verbose=False,
        )

        mu_bl_excess_w, M_w = black_litterman(
            cov_matrix=cov,
            returns=train,
            risk_free_rate=risk_free_rate,
            delta=bl_delta,
            tau=bl_tau,
            P=P_eff,
            Q=Q_eff,
            Omega=Omega_eff,
            view_confidence_scale=bl_view_confidence_scale,
            return_posterior_cov=True,
            w_market=w_market,
        )

        # Conversion en rendement absolu
        mu_bl_w = mu_bl_excess_w + risk_free_rate

        mu_w = shrink_mu(mu_bl_w, mu_shrink_factor)
        vol_cov_w = (cov + M_w) if use_posterior_cov else None

        try:
            w = optimize_portfolio(
                returns=train, cov_matrix=cov, target_volatility=target_volatility,
                risk_free_rate=risk_free_rate, objective=objective,
                max_weight=max_weight, min_weight=min_weight,
                mu_override=mu_w, vol_cov_matrix=vol_cov_w,
            )["weights"]
        except (RuntimeError, ValueError):
            w = np.ones(n) / n

        turnover = float(np.sum(np.abs(w - prev_weights)))
        cost     = turnover * transaction_cost_bps / 10_000.0
        test_lr  = test.values @ w
        if transaction_cost_bps > 0.0 and len(test_lr) > 0:
            test_lr    = test_lr.copy()
            test_lr[0] -= cost

        strat.extend(test_lr)
        eqw.extend(test.values @ (np.ones(n) / n))
        idx.extend(test.index)
        w_hist.append((test.index[0], w, turnover, cost))

        gf           = np.exp(test.values.sum(axis=0))
        drifted      = w * gf
        prev_weights = drifted / drifted.sum()
        start       += rebalance

    oos = pd.DataFrame({"strategie": strat, "equipondere": eqw},
                       index=pd.DatetimeIndex(idx))
    return oos, w_hist


def run_backtest(
    tickers, target_volatility, period="10y", risk_free_rate=0.04,
    objective="target_vol", max_weight=1.0, min_weight=0.0,
    bl_delta=2.5, bl_tau=0.05, bl_views_P=None, bl_views_Q=None,
    bl_views_Omega=None, bl_view_confidence_scale=1.0,
    mu_shrink_factor=1.0, use_posterior_cov=False,
    transaction_cost_bps=10.0,
    use_momentum_views=True, momentum_lookback=252, momentum_skip=21,
    momentum_signal_scale=0.07, momentum_confidence_scale=1.5,
    z_clip=2.0, q_cap_momentum=0.15,
    use_analyst_views=True, analyst_q_cap=0.20, analyst_confidence_scale=2.5,
    use_ml_views=True, ml_forward_window=21, ml_ridge_alpha=1.0,
    ml_signal_scale=0.25, ml_q_cap=0.12, ml_confidence_scale=2.0,
    w_momentum=0.40, w_analyst=0.35, w_ml=0.25,
    view_confidence_scale_fused=1.5,
):
    returns = fetch_returns(tickers, period)
    # [CORRECTION 21] Réalignement avant tout calcul dérivé de l'ordre des tickers.
    tickers  = list(returns.columns)
    w_market = fetch_market_cap_weights(tickers)
    kwargs   = dict(
        use_momentum_views=use_momentum_views, momentum_lookback=momentum_lookback,
        momentum_skip=momentum_skip, momentum_signal_scale=momentum_signal_scale,
        momentum_confidence_scale=momentum_confidence_scale,
        z_clip=z_clip, q_cap_momentum=q_cap_momentum,
        use_analyst_views=use_analyst_views, analyst_q_cap=analyst_q_cap,
        analyst_confidence_scale=analyst_confidence_scale,
        use_ml_views=use_ml_views, ml_forward_window=ml_forward_window,
        ml_ridge_alpha=ml_ridge_alpha, ml_signal_scale=ml_signal_scale,
        ml_q_cap=ml_q_cap, ml_confidence_scale=ml_confidence_scale,
        w_momentum=w_momentum, w_analyst=w_analyst, w_ml=w_ml,
        view_confidence_scale_fused=view_confidence_scale_fused,
    )
    oos, w_hist = backtest_oos(
        returns=returns, target_volatility=target_volatility,
        risk_free_rate=risk_free_rate, objective=objective,
        lookback=504, rebalance=63, max_weight=max_weight, min_weight=min_weight,
        bl_delta=bl_delta, bl_tau=bl_tau, bl_views_P=bl_views_P,
        bl_views_Q=bl_views_Q, bl_views_Omega=bl_views_Omega,
        bl_view_confidence_scale=bl_view_confidence_scale,
        mu_shrink_factor=mu_shrink_factor, use_posterior_cov=use_posterior_cov,
        transaction_cost_bps=transaction_cost_bps, w_market=w_market, **kwargs,
    )
    m_s        = perf_metrics(oos["strategie"], risk_free_rate)
    m_e        = perf_metrics(oos["equipondere"], risk_free_rate)
    avg_to     = np.mean([t for _, _, t, _ in w_hist])
    total_cost = sum(c for _, _, _, c in w_hist)

    print("\n" + "═" * 72)
    print(f"   BACKTEST OOS  (objectif : {objective}, coût : {transaction_cost_bps:.0f} bps/rebal.)")
    print("═" * 72)
    print(f"\n  {'':<14}{'CAGR':>9}{'Rdt log':>10}{'Volat.':>10}{'Sharpe':>9}{'Max DD':>9}")
    for name, m in (("Stratégie", m_s), ("Équipondéré", m_e)):
        print(f"  {name:<14}{m['cagr']*100:>8.2f}%{m['rendement_log']*100:>9.2f}%"
              f"{m['volatilite']*100:>9.2f}%{m['sharpe']:>9.3f}{m['max_drawdown']*100:>8.1f}%")
    print(f"\n  Turnover moyen : {avg_to*100:.1f}%")
    print(f"  Coût cumulé    : {total_cost*100:.2f}%")
    print(f"\n  Sharpe OOS actif seul (buy & hold) :")
    oos_period = returns.loc[oos.index]
    for c in returns.columns:
        print(f"    {c:<12} {perf_metrics(oos_period[c], risk_free_rate)['sharpe']:>7.3f}")
    print("═" * 72 + "\n")
    return oos


# ─────────────────────────────────────────────────────────────────────────────
# 13. COMPARAISON MULTI-STRATÉGIES
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest_compare(
    tickers, target_volatility, period="10y", risk_free_rate=0.04,
    max_weight=1.0, min_weight=0.0, transaction_cost_bps=10.0,
    bl_delta=2.5, bl_tau=0.05, bl_views_P=None, bl_views_Q=None,
    bl_views_Omega=None, bl_view_confidence_scale=1.0,
    mu_shrink_factor=1.0, use_posterior_cov=False,
    use_momentum_views=True, momentum_lookback=252, momentum_skip=21,
    momentum_signal_scale=0.07, momentum_confidence_scale=1.5,
    z_clip=2.0, q_cap_momentum=0.15,
    use_analyst_views=True, analyst_q_cap=0.20, analyst_confidence_scale=2.5,
    use_ml_views=True, ml_forward_window=21, ml_ridge_alpha=1.0,
    ml_signal_scale=0.25, ml_q_cap=0.12, ml_confidence_scale=2.0,
    w_momentum=0.40, w_analyst=0.35, w_ml=0.25,
    view_confidence_scale_fused=1.5,
):
    returns = fetch_returns(tickers, period)
    tickers  = list(returns.columns)
    w_market = fetch_market_cap_weights(tickers)

    bl_kwargs = dict(
        bl_delta=bl_delta, bl_tau=bl_tau, bl_views_P=bl_views_P,
        bl_views_Q=bl_views_Q, bl_views_Omega=bl_views_Omega,
        bl_view_confidence_scale=bl_view_confidence_scale,
        mu_shrink_factor=mu_shrink_factor, use_posterior_cov=use_posterior_cov,
        use_momentum_views=use_momentum_views, momentum_lookback=momentum_lookback,
        momentum_skip=momentum_skip, momentum_signal_scale=momentum_signal_scale,
        momentum_confidence_scale=momentum_confidence_scale,
        z_clip=z_clip, q_cap_momentum=q_cap_momentum,
        use_analyst_views=use_analyst_views, analyst_q_cap=analyst_q_cap,
        analyst_confidence_scale=analyst_confidence_scale,
        use_ml_views=use_ml_views, ml_forward_window=ml_forward_window,
        ml_ridge_alpha=ml_ridge_alpha, ml_signal_scale=ml_signal_scale,
        ml_q_cap=ml_q_cap, ml_confidence_scale=ml_confidence_scale,
        w_momentum=w_momentum, w_analyst=w_analyst, w_ml=w_ml,
        view_confidence_scale_fused=view_confidence_scale_fused,
        w_market=w_market,
    )
    no_views_kwargs = dict(w_market=w_market)

    strategies = {
        "BL target_vol": dict(objective="target_vol", **bl_kwargs),
        "BL max_sharpe": dict(objective="max_sharpe", **bl_kwargs),
        "min_variance":  dict(objective="min_variance",  **no_views_kwargs),
        "risk_parity":   dict(objective="risk_parity",   **no_views_kwargs),
        "hrp":           dict(objective="hrp",           **no_views_kwargs),
    }

    rows, first_oos_index = [], None
    for name, kwargs in strategies.items():
        oos, w_hist = backtest_oos(
            returns=returns, target_volatility=target_volatility,
            risk_free_rate=risk_free_rate, lookback=504, rebalance=63,
            max_weight=max_weight, min_weight=min_weight,
            transaction_cost_bps=transaction_cost_bps, **kwargs,
        )
        m      = perf_metrics(oos["strategie"], risk_free_rate)
        avg_to = float(np.mean([t for _, _, t, _ in w_hist]))
        rows.append((name, m["cagr"], m["rendement_log"], m["volatilite"],
                     m["sharpe"], m["max_drawdown"], avg_to))
        if first_oos_index is None:
            first_oos_index = oos.index

    eqw = (returns.loc[first_oos_index] @ (np.ones(returns.shape[1]) / returns.shape[1])).values
    m_e = perf_metrics(pd.Series(eqw, index=first_oos_index), risk_free_rate)
    rows.append(("Équipondéré", m_e["cagr"], m_e["rendement_log"], m_e["volatilite"],
                 m_e["sharpe"], m_e["max_drawdown"], 0.0))
    rows.sort(key=lambda r: r[4], reverse=True)

    print("\n" + "═" * 86)
    print(f"   COMPARAISON MULTI-STRATÉGIES — OOS  "
          f"(coût : {transaction_cost_bps:.0f} bps/rebal., vol cible : {target_volatility*100:.1f}%)")
    print("═" * 86)
    print(f"\n  {'Stratégie':<16}{'CAGR':>9}{'Rdt log':>10}{'Volat.':>10}"
          f"{'Sharpe':>9}{'Max DD':>9}{'Turnover':>11}")
    print("  " + "─" * 84)
    for name, cagr, rlog, vol, sharpe, mdd, to in rows:
        print(f"  {name:<16}{cagr*100:>8.2f}%{rlog*100:>9.2f}%{vol*100:>9.2f}%"
              f"{sharpe:>9.3f}{mdd*100:>8.1f}%{to*100:>10.1f}%")
    print("═" * 86)
    print("  Classement par Sharpe OOS décroissant.")
    print("  HRP/risk_parity ignorent mu — Sharpe dépend uniquement de la qualité de Σ.")
    print(f"  Views BL = fusion momentum ({w_momentum*100:.0f}%) + "
          f"analyst ({w_analyst*100:.0f}%) + ML Ridge ({w_ml*100:.0f}%).")
    print("═" * 86 + "\n")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TICKERS = [
        "NVDA",       # Nvidia — USD, semi-conducteurs 
        "GOOGL",      # Alphabet — USD, tech 
        "TTE.PA",     # TotalEnergies — EUR, énergie, Euronext Paris
        "OR.PA",      # L'Oréal — EUR, consommation, Euronext Paris
        "SIE.DE",     # Siemens — EUR, industrie, Xetra
        "SLV",        # iShares Silver Trust — USD, métal précieux (comme GLD : pas de market cap → fallback médiane)
        "MSFT",       # Microsoft — USD
        "ASML.AS",    # ASML — EUR, Euronext Amsterdam
        "IWDA.AS",    # iShares MSCI World — EUR
        "MC.PA",      # LVMH — EUR, Euronext Paris
        "SAP.DE",     # SAP — EUR, Xetra
        "BTC-USD", 

    TOTAL_AMOUNT = 3_000.0
    TARGET_VOL   = 0.15
    PERIOD       = "10y"
    RF           = 0.04
    OBJECTIVE    = "target_vol"
    MAX_W        = 0.3
    MIN_W        = 0.0
    COST_BPS     = 10.0

    BL_DELTA     = 4.0
    BL_TAU       = 0.1
    BL_CONF      = 1.0
    MU_SHRINK    = 1.0
    USE_POST_COV = False

    USE_MOM      = True
    MOM_LOOKBACK = 252
    MOM_SKIP     = 21
    MOM_SCALE    = 0.07   # [CORRECTION 22] était 0.3 → saturait le cap à |z|>0.5
    MOM_CONF     = 0.8
    Z_CLIP       = 2.0
    Q_CAP_MOM    = 0.15

    USE_ANALYST  = True   # les appels seront court-circuités si W_ANA=0
    ANA_Q_CAP    = 0.20
    ANA_CONF     = 2.5

    USE_ML       = True
    ML_FW        = 21
    ML_ALPHA     = 1.0
    ML_SCALE     = 0.06
    ML_Q_CAP     = 0.12
    ML_CONF      = 1.3

    W_MOM        = 0.70
    W_ANA        = 0.0   # 0 → appels analyst automatiquement désactivés
    W_ML         = 0.30
    CONF_FUSED   = 0.8

    BL_VIEWS_P     = None
    BL_VIEWS_Q     = None
    BL_VIEWS_OMEGA = None

    shared = dict(
        use_momentum_views=USE_MOM, momentum_lookback=MOM_LOOKBACK,
        momentum_skip=MOM_SKIP, momentum_signal_scale=MOM_SCALE,
        momentum_confidence_scale=MOM_CONF, z_clip=Z_CLIP, q_cap_momentum=Q_CAP_MOM,
        use_analyst_views=USE_ANALYST, analyst_q_cap=ANA_Q_CAP,
        analyst_confidence_scale=ANA_CONF,
        use_ml_views=USE_ML, ml_forward_window=ML_FW, ml_ridge_alpha=ML_ALPHA,
        ml_signal_scale=ML_SCALE, ml_q_cap=ML_Q_CAP, ml_confidence_scale=ML_CONF,
        w_momentum=W_MOM, w_analyst=W_ANA, w_ml=W_ML,
        view_confidence_scale_fused=CONF_FUSED,
    )

    run_optimizer(
        tickers=TICKERS, total_amount=TOTAL_AMOUNT, target_volatility=TARGET_VOL,
        period=PERIOD, risk_free_rate=RF, objective=OBJECTIVE,
        max_weight=MAX_W, min_weight=MIN_W, bl_delta=BL_DELTA, bl_tau=BL_TAU,
        bl_views_P=BL_VIEWS_P, bl_views_Q=BL_VIEWS_Q, bl_views_Omega=BL_VIEWS_OMEGA,
        bl_view_confidence_scale=BL_CONF, mu_shrink_factor=MU_SHRINK,
        use_posterior_cov=USE_POST_COV, **shared,
    )

    run_backtest(
        tickers=TICKERS, target_volatility=TARGET_VOL, period=PERIOD,
        risk_free_rate=RF, objective=OBJECTIVE, max_weight=MAX_W, min_weight=MIN_W,
        bl_delta=BL_DELTA, bl_tau=BL_TAU, bl_views_P=BL_VIEWS_P,
        bl_views_Q=BL_VIEWS_Q, bl_views_Omega=BL_VIEWS_OMEGA,
        bl_view_confidence_scale=BL_CONF, mu_shrink_factor=MU_SHRINK,
        use_posterior_cov=USE_POST_COV, transaction_cost_bps=COST_BPS, **shared,
    )

    run_backtest_compare(
        tickers=TICKERS, target_volatility=TARGET_VOL, period=PERIOD,
        risk_free_rate=RF, max_weight=MAX_W, min_weight=MIN_W,
        transaction_cost_bps=COST_BPS, bl_delta=BL_DELTA, bl_tau=BL_TAU,
        bl_views_P=BL_VIEWS_P, bl_views_Q=BL_VIEWS_Q, bl_views_Omega=BL_VIEWS_OMEGA,
        bl_view_confidence_scale=BL_CONF, mu_shrink_factor=MU_SHRINK,
        use_posterior_cov=USE_POST_COV, **shared,
    )
