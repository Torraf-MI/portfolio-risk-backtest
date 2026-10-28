"""Strategie di allocazione da confrontare nel backtest.

Ogni strategia è una funzione che riceve i prezzi disponibili fino alla data di ribilanciamento e
restituisce i pesi obiettivo. La somma può essere inferiore a 1: il resto resta in liquidità a rendimento
nullo (le strategie che riducono il rischio usano l'asset monetario, se c'è).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import optimize

from .dati import classe

Strategia = Callable[[pd.DataFrame], pd.Series]


def _per_classe(colonne: list[str]) -> dict[str | None, list[str]]:
    gruppi: dict[str | None, list[str]] = {}
    for c in colonne:
        gruppi.setdefault(classe(c), []).append(c)
    return gruppi


def _monetario(colonne: list[str]) -> str | None:
    return next(iter(_per_classe(colonne).get("liquidita", [])), None)


def covarianza(prezzi: pd.DataFrame, finestra: int = 252) -> pd.DataFrame:
    """Covarianza annua dell'ultimo anno, con shrinkage verso la diagonale (stima più stabile)."""
    r = prezzi.pct_change().iloc[-finestra:].dropna()
    s = r.cov() * 252
    d = pd.DataFrame(np.diag(np.diag(s)), index=s.index, columns=s.columns)
    return 0.8 * s + 0.2 * d


def costante(pesi: pd.Series | dict) -> Strategia:
    """Pesi fissi, ripristinati a ogni ribilanciamento (ad esempio il portafoglio attuale del cliente)."""
    w = pd.Series(pesi, dtype=float)
    return lambda prezzi: w.reindex(prezzi.columns).fillna(0.0)


def bilanciato_60_40(prezzi: pd.DataFrame) -> pd.Series:
    """60% azioni e 40% obbligazioni, a pesi uguali all'interno di ciascuna parte."""
    g = _per_classe(list(prezzi.columns))
    azioni = g.get("azioni", [])
    obbligazioni = g.get("governativi", []) + g.get("corporate", [])
    if not azioni or not obbligazioni:
        raise ValueError("La strategia 60/40 richiede almeno un asset azionario e uno obbligazionario")
    w = pd.Series(0.0, index=prezzi.columns)
    w[azioni] = 0.6 / len(azioni)
    w[obbligazioni] = 0.4 / len(obbligazioni)
    return w


def pesi_uguali(prezzi: pd.DataFrame) -> pd.Series:
    """Stesso peso a ogni asset rischioso (la liquidità è esclusa)."""
    m = _monetario(list(prezzi.columns))
    rischiosi = [c for c in prezzi.columns if c != m]
    return pd.Series(1 / len(rischiosi), index=rischiosi).reindex(prezzi.columns).fillna(0.0)


def risk_parity(prezzi: pd.DataFrame) -> pd.Series:
    """Equal Risk Contribution: ogni asset rischioso contribuisce allo stesso modo alla volatilità.

    Risolve il problema convesso di Spinu: min ½·wᵀΣw − (1/n)·Σ log wᵢ, poi normalizza i pesi a 1.
    """
    m = _monetario(list(prezzi.columns))
    rischiosi = [c for c in prezzi.columns if c != m]
    cov = covarianza(prezzi[rischiosi]).to_numpy()
    n = len(rischiosi)
    scala = np.sqrt(np.diag(cov))
    c = cov / np.outer(scala, scala)  # risolve sulla correlazione per un problema ben condizionato

    def obiettivo(y: np.ndarray) -> tuple[float, np.ndarray]:
        cy = c @ y
        return 0.5 * y @ cy - np.log(y).sum() / n, cy - 1 / (n * y)

    ris = optimize.minimize(obiettivo, np.full(n, 1 / np.sqrt(n)), jac=True, method="L-BFGS-B",
                            bounds=[(1e-8, None)] * n)
    w = ris.x / scala
    return pd.Series(w / w.sum(), index=rischiosi).reindex(prezzi.columns).fillna(0.0)


def minima_varianza(prezzi: pd.DataFrame, peso_massimo: float = 0.4) -> pd.Series:
    """Portafoglio a minima volatilità, solo posizioni lunghe, con un tetto per asset."""
    m = _monetario(list(prezzi.columns))
    rischiosi = [c for c in prezzi.columns if c != m]
    cov = covarianza(prezzi[rischiosi]).to_numpy()
    n = len(rischiosi)
    tetto = max(peso_massimo, 1 / n)
    ris = optimize.minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), jac=lambda w: 2 * cov @ w, method="SLSQP",
                            bounds=[(0, tetto)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w = np.clip(ris.x, 0, None)
    return pd.Series(w / w.sum(), index=rischiosi).reindex(prezzi.columns).fillna(0.0)


def volatilita_obiettivo(base: Strategia = bilanciato_60_40, obiettivo: float = 0.08, lam: float = 0.97) -> Strategia:
    """Riduce l'esposizione della strategia di base quando la volatilità prevista supera l'obiettivo.

    La volatilità è stimata con una media mobile esponenziale dei rendimenti giornalieri del portafoglio di
    base; l'esposizione non supera mai il 100% (niente leva) e la parte tolta va in monetario.
    """

    def strategia(prezzi: pd.DataFrame) -> pd.Series:
        w = base(prezzi)
        r = prezzi.pct_change().iloc[-252:].fillna(0.0) @ w
        pesi_tempo = lam ** np.arange(len(r))[::-1]
        vol = float(np.sqrt(np.sum(pesi_tempo * r.to_numpy() ** 2) / pesi_tempo.sum() * 252))
        esposizione = min(1.0, obiettivo / vol) if vol > 0 else 1.0
        w = w * esposizione
        m = _monetario(list(prezzi.columns))
        if m is not None:
            w[m] += 1 - esposizione
        return w

    return strategia


def trend_following(base: Strategia = bilanciato_60_40, giorni: int = 200) -> Strategia:
    """Filtro di tendenza (Faber): un asset resta in portafoglio solo se è sopra la media mobile a 200 giorni.

    Il peso degli asset esclusi passa al monetario o, se manca, resta in liquidità.
    """

    def strategia(prezzi: pd.DataFrame) -> pd.Series:
        w = base(prezzi)
        sopra = prezzi.iloc[-1] > prezzi.iloc[-giorni:].mean()
        tolto = float(w[~sopra].sum())
        w = w.where(sopra, 0.0)
        m = _monetario(list(prezzi.columns))
        if m is not None:
            w[m] += tolto
        return w

    return strategia


def strategie_standard(pesi_cliente: pd.Series | None = None) -> dict[str, Strategia]:
    """Le strategie confrontate nel report, con il portafoglio del cliente in testa se indicato."""
    s: dict[str, Strategia] = {}
    if pesi_cliente is not None:
        s["Portafoglio attuale"] = costante(pesi_cliente)
    s |= {
        "60/40": bilanciato_60_40,
        "Pesi uguali": pesi_uguali,
        "Risk parity": risk_parity,
        "Minima varianza": minima_varianza,
        "60/40 a volatilità obiettivo": volatilita_obiettivo(),
        "60/40 con filtro di tendenza": trend_following(),
    }
    return s
