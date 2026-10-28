"""Misure di rischio: VaR e CVaR con sei metodi, volatilità EWMA e GARCH(1,1), contributi al rischio.

Convenzioni: i rendimenti sono semplici e giornalieri; VaR e CVaR sono **perdite** espresse come numeri
positivi (0,021 = perdita del 2,1%). ``livello`` è il livello di confidenza (0,99 = 99%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate, optimize, signal, stats

METODI = ["Storico", "Normale", "Cornish-Fisher", "t di Student", "EWMA (RiskMetrics)", "GARCH(1,1)"]


def rendimenti(prezzi: pd.DataFrame) -> pd.DataFrame:
    """Rendimenti semplici giornalieri."""
    return prezzi.pct_change().dropna(how="all").fillna(0.0)


def rendimenti_portafoglio(r: pd.DataFrame, pesi: pd.Series) -> pd.Series:
    """Rendimento giornaliero di un portafoglio a pesi costanti (ribilanciato ogni giorno)."""
    return (r[pesi.index] @ pesi).rename("Portafoglio")


# --------------------------------------------------------------------------- #
# VaR e CVaR
# --------------------------------------------------------------------------- #
def var_storico(r: pd.Series | np.ndarray, livello: float = 0.99) -> float:
    """Quantile empirico delle perdite."""
    return float(-np.quantile(np.asarray(r), 1 - livello))


def cvar_storico(r: pd.Series | np.ndarray, livello: float = 0.99) -> float:
    """Perdita media nei giorni peggiori del VaR (Expected Shortfall)."""
    x = np.asarray(r)
    soglia = np.quantile(x, 1 - livello)
    return float(-x[x <= soglia].mean())


def var_normale(mu: float, sigma: float, livello: float = 0.99) -> float:
    return float(-(mu + sigma * stats.norm.ppf(1 - livello)))


def cvar_normale(mu: float, sigma: float, livello: float = 0.99) -> float:
    z = stats.norm.ppf(1 - livello)
    return float(-mu + sigma * stats.norm.pdf(z) / (1 - livello))


def quantile_cornish_fisher(p: float, asimmetria: float, curtosi_eccesso: float) -> float:
    """Quantile standardizzato corretto per asimmetria e curtosi (espansione di Cornish-Fisher)."""
    z, s, k = stats.norm.ppf(p), asimmetria, curtosi_eccesso
    return float(z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36)


def var_cornish_fisher(r: pd.Series | np.ndarray, livello: float = 0.99) -> float:
    x = np.asarray(r)
    s, k = stats.skew(x), stats.kurtosis(x)
    return float(-(x.mean() + x.std(ddof=1) * quantile_cornish_fisher(1 - livello, s, k)))


def cvar_cornish_fisher(r: pd.Series | np.ndarray, livello: float = 0.99) -> float:
    """Media dei VaR di Cornish-Fisher oltre il livello (integrazione numerica sulla coda)."""
    x = np.asarray(r)
    mu, sigma, s, k = x.mean(), x.std(ddof=1), stats.skew(x), stats.kurtosis(x)
    coda, _ = integrate.quad(lambda p: quantile_cornish_fisher(p, s, k), 1e-9, 1 - livello)
    return float(-(mu + sigma * coda / (1 - livello)))


def var_t(r: pd.Series | np.ndarray, livello: float = 0.99) -> tuple[float, float, dict[str, float]]:
    """VaR e CVaR con una t di Student stimata per massima verosimiglianza; restituisce anche i parametri."""
    x = np.asarray(r)
    gl, loc, scala = stats.t.fit(x)
    gl = max(gl, 2.05)  # con meno di 2 gradi di libertà la varianza non esiste
    p = 1 - livello
    q = stats.t.ppf(p, gl)
    var = -(loc + scala * q)
    cvar = -loc + scala * (gl + q**2) / (gl - 1) * stats.t.pdf(q, gl) / p
    return float(var), float(cvar), {"gradi_liberta": float(gl), "loc": float(loc), "scala": float(scala)}


# --------------------------------------------------------------------------- #
# Volatilità condizionata
# --------------------------------------------------------------------------- #
def volatilita_ewma(r: pd.Series, lam: float = 0.94, inizio: int = 30) -> pd.Series:
    """Previsione EWMA della volatilità per il giorno successivo (RiskMetrics, λ = 0,94).

    Il valore alla data *t* usa i rendimenti fino a *t* compreso ed è la previsione per *t+1*.
    """
    x = np.asarray(r, dtype=float)
    iniziale = np.mean(x[:inizio] ** 2)
    # var[t] = λ·var[t−1] + (1−λ)·r²[t], calcolata come filtro ricorsivo
    var, _ = signal.lfilter([1 - lam], [1, -lam], x**2, zi=[lam * iniziale])
    return pd.Series(np.sqrt(var), index=getattr(r, "index", None), name="Volatilità EWMA")


@dataclass
class Garch:
    """GARCH(1,1): σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁, stimato per massima verosimiglianza."""

    omega: float
    alpha: float
    beta: float
    mu: float

    @property
    def persistenza(self) -> float:
        return self.alpha + self.beta

    @property
    def volatilita_lungo_periodo(self) -> float:
        """Volatilità annua verso cui tende il modello."""
        return float(np.sqrt(self.omega / (1 - self.persistenza) * 252))

    def varianze(self, r: np.ndarray, iniziale: float | None = None) -> np.ndarray:
        """Varianze condizionate; l'elemento *t* è la previsione per *t* fatta con i dati fino a *t−1*.

        L'array ha un elemento in più: l'ultimo è la previsione per il giorno successivo all'ultimo dato.
        La varianza iniziale, se non indicata, è quella del campione.
        """
        e = np.asarray(r, dtype=float) - self.mu
        v0 = float(np.var(e)) if iniziale is None else iniziale
        return _ricorsione(self.omega, self.alpha, self.beta, np.append(e, 0.0), v0)


def _ricorsione(omega: float, alpha: float, beta: float, e: np.ndarray, v0: float) -> np.ndarray:
    """s2[0] = v0, s2[t] = ω + α·e²[t−1] + β·s2[t−1], vettorizzata con un filtro lineare."""
    ingresso = omega + alpha * e[:-1] ** 2
    coda, _ = signal.lfilter([1.0], [1.0, -beta], ingresso, zi=[beta * v0])
    return np.concatenate([[v0], coda])


def _negloglik(theta: np.ndarray, e: np.ndarray, v0: float) -> float:
    omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10
    s2 = _ricorsione(omega, alpha, beta, e, v0)
    return float(0.5 * np.sum(np.log(s2) + e**2 / s2))


def stima_garch(r: pd.Series | np.ndarray) -> Garch:
    """Stima di un GARCH(1,1) gaussiano con ``scipy.optimize`` (Nelder-Mead, vincoli di stazionarietà)."""
    x = np.asarray(r, dtype=float)
    mu = float(x.mean())
    e = x - mu
    v = float(np.var(e))
    partenza = np.array([v * 0.05, 0.08, 0.87])
    # Ottimizza su parametri riscalati: omega è di ordini di grandezza più piccolo di alpha e beta
    scala = np.array([v, 1.0, 1.0])
    ris = optimize.minimize(lambda th: _negloglik(th * scala, e, v), partenza / scala, method="Nelder-Mead",
                            options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 4000})
    omega, alpha, beta = ris.x * scala
    return Garch(omega=float(omega), alpha=float(alpha), beta=float(beta), mu=mu)


# --------------------------------------------------------------------------- #
# Sintesi su un orizzonte
# --------------------------------------------------------------------------- #
def tabella_var(r: pd.Series, livello: float = 0.99, orizzonte: int = 1) -> pd.DataFrame:
    """VaR e CVaR del portafoglio con tutti i metodi, scalati all'orizzonte con la regola della radice.

    I metodi EWMA e GARCH usano la volatilità prevista per domani, quindi rispondono alle condizioni di
    mercato attuali; gli altri descrivono la distribuzione dell'intero campione.
    """
    x = r.to_numpy()
    mu, sigma = x.mean(), x.std(ddof=1)
    vt, cvt, _ = var_t(x, livello)
    s_ewma = float(volatilita_ewma(r).iloc[-1])
    g = stima_garch(x)
    s_garch = float(np.sqrt(g.varianze(x)[-1]))
    righe = {
        "Storico": (var_storico(x, livello), cvar_storico(x, livello), sigma),
        "Normale": (var_normale(mu, sigma, livello), cvar_normale(mu, sigma, livello), sigma),
        "Cornish-Fisher": (var_cornish_fisher(x, livello), cvar_cornish_fisher(x, livello), sigma),
        "t di Student": (vt, cvt, sigma),
        "EWMA (RiskMetrics)": (var_normale(0, s_ewma, livello), cvar_normale(0, s_ewma, livello), s_ewma),
        "GARCH(1,1)": (var_normale(0, s_garch, livello), cvar_normale(0, s_garch, livello), s_garch),
    }
    t = pd.DataFrame(righe, index=["VaR", "CVaR", "Volatilità giornaliera"]).T
    t[["VaR", "CVaR"]] *= np.sqrt(orizzonte)
    t["Volatilità annua"] = t.pop("Volatilità giornaliera") * np.sqrt(252)
    return t


# --------------------------------------------------------------------------- #
# Contributi al rischio
# --------------------------------------------------------------------------- #
def contributi_volatilita(r: pd.DataFrame, pesi: pd.Series) -> pd.Series:
    """Contributo di ogni asset alla volatilità annua del portafoglio (somma = volatilità)."""
    cov = r[pesi.index].cov().to_numpy() * 252
    w = pesi.to_numpy()
    sigma = np.sqrt(w @ cov @ w)
    return pd.Series(w * (cov @ w) / sigma, index=pesi.index, name="Contributo alla volatilità")


def contributi_cvar(r: pd.DataFrame, pesi: pd.Series, livello: float = 0.99) -> pd.Series:
    """Contributo di ogni asset al CVaR storico: perdita media dell'asset nei giorni di coda del portafoglio.

    È la scomposizione di Eulero dell'Expected Shortfall, quindi i contributi sommano al CVaR.
    """
    rp = r[pesi.index] @ pesi
    coda = rp <= np.quantile(rp, 1 - livello)
    return (-(r.loc[coda, pesi.index] * pesi).mean()).rename("Contributo al CVaR")


def correlazione_mobile(r: pd.DataFrame, a: str, b: str, finestra: int = 252) -> pd.Series:
    """Correlazione su finestra mobile tra due asset (ad esempio azioni e obbligazioni)."""
    return r[a].rolling(finestra).corr(r[b]).dropna().rename(f"{a} / {b}")
