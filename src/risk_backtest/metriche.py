"""Metriche di rendimento e rischio delle strategie, con test statistici sulle differenze."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .rischio import cvar_storico, var_storico


def drawdown(r: pd.Series) -> pd.Series:
    """Perdita dal massimo precedente, giorno per giorno."""
    valore = (1 + r).cumprod()
    return valore / np.maximum(valore.cummax(), 1.0) - 1


def durata_massima_drawdown(r: pd.Series) -> int:
    """Numero massimo di giorni di borsa consecutivi sotto il massimo precedente."""
    sotto = (drawdown(r) < 0).to_numpy()
    massimo = corrente = 0
    for s in sotto:
        corrente = corrente + 1 if s else 0
        massimo = max(massimo, corrente)
    return massimo


def cagr(r: pd.Series) -> float:
    anni = len(r) / 252
    return float((1 + r).prod() ** (1 / anni) - 1)


def sharpe(r: pd.Series, risk_free: float = 0.0) -> float:
    eccesso = r - risk_free / 252
    return float(eccesso.mean() / eccesso.std() * np.sqrt(252))


def sortino(r: pd.Series, risk_free: float = 0.0) -> float:
    eccesso = r - risk_free / 252
    ribasso = np.sqrt(np.mean(np.minimum(eccesso, 0) ** 2))
    return float(eccesso.mean() / ribasso * np.sqrt(252))


def sharpe_probabilistico(r: pd.Series, soglia: float = 0.0, risk_free: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey e López de Prado): probabilità che lo Sharpe vero superi ``soglia``.

    Tiene conto della lunghezza del campione e di asimmetria e curtosi dei rendimenti. Sharpe annui.
    """
    eccesso = (r - risk_free / 252).to_numpy()
    n = len(eccesso)
    sr = eccesso.mean() / eccesso.std(ddof=1)  # giornaliero
    s, k = stats.skew(eccesso), stats.kurtosis(eccesso, fisher=False)
    soglia_g = soglia / np.sqrt(252)
    errore = np.sqrt((1 - s * sr + (k - 1) / 4 * sr**2) / (n - 1))
    return float(stats.norm.cdf((sr - soglia_g) / errore))


def sintesi(r: pd.Series, risk_free: float = 0.0, livello: float = 0.95) -> dict[str, float]:
    """Indicatori principali di una serie di rendimenti giornalieri."""
    mensili = (1 + r).resample("ME").prod() - 1
    dd = drawdown(r).min()
    c = cagr(r)
    return {
        "Rendimento annuo": c,
        "Volatilità": float(r.std() * np.sqrt(252)),
        "Sharpe": sharpe(r, risk_free),
        "Sortino": sortino(r, risk_free),
        "Massima perdita": float(dd),
        "Calmar": float(c / -dd) if dd < 0 else np.nan,
        "Mesi sotto il massimo": durata_massima_drawdown(r) / 21,
        f"VaR {livello:.0%} giornaliero": var_storico(r, livello),
        f"CVaR {livello:.0%} giornaliero": cvar_storico(r, livello),
        "Mese peggiore": float(mensili.min()),
        "Mesi positivi": float((mensili > 0).mean()),
        "Asimmetria": float(stats.skew(r)),
        "Curtosi in eccesso": float(stats.kurtosis(r)),
        "Probabilità Sharpe > 0": sharpe_probabilistico(r, 0.0, risk_free),
    }


def tabella(rendimenti: dict[str, pd.Series] | pd.DataFrame, risk_free: float = 0.0) -> pd.DataFrame:
    """Una riga di :func:`sintesi` per ogni strategia."""
    return pd.DataFrame({nome: sintesi(r, risk_free) for nome, r in rendimenti.items()}).T


def alfa_beta(r: pd.Series, benchmark: pd.Series, risk_free: float = 0.0) -> dict[str, float]:
    """Regressione dei rendimenti in eccesso mensili sul benchmark, con errori robusti (Newey-West).

    L'alfa è annualizzato; il p-value dice se è statisticamente diverso da zero.
    """
    dati = pd.concat([r, benchmark], axis=1, keys=["s", "b"]).dropna()
    mensili = (1 + dati).resample("ME").prod() - 1 - risk_free / 12
    modello = sm.OLS(mensili["s"], sm.add_constant(mensili["b"])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return {
        "Alfa annuo": float(modello.params["const"] * 12),
        "p-value alfa": float(modello.pvalues["const"]),
        "Beta": float(modello.params["b"]),
        "R²": float(modello.rsquared),
        "Tracking error": float((mensili["s"] - mensili["b"]).std() * np.sqrt(12)),
    }


def rendimenti_annuali(rendimenti: dict[str, pd.Series]) -> pd.DataFrame:
    """Rendimento di ogni anno solare per ogni strategia."""
    df = pd.DataFrame(rendimenti)
    annui = (1 + df).resample("YE").prod() - 1
    annui.index = annui.index.year
    return annui
