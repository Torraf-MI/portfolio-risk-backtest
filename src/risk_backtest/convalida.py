"""Backtest del VaR: previsioni giorno per giorno fuori campione e test statistici sulle violazioni.

Ogni previsione usa solo i dati disponibili il giorno prima, come avverrebbe nella gestione reale.
Le violazioni sono i giorni in cui la perdita effettiva supera il VaR previsto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .rischio import Garch, quantile_cornish_fisher, stima_garch, volatilita_ewma

METODI_CONVALIDA = ["Storico", "Normale", "Cornish-Fisher", "EWMA (RiskMetrics)", "GARCH(1,1)"]


def previsioni_var(
    r: pd.Series,
    livello: float = 0.99,
    finestra: int = 500,
    ristima_garch: int = 250,
) -> pd.DataFrame:
    """VaR a un giorno previsto per ogni data con i metodi di :data:`METODI_CONVALIDA`.

    I metodi su campione usano una finestra mobile di ``finestra`` giorni; il GARCH è ristimato ogni
    ``ristima_garch`` giorni sui dati della finestra e filtrato giorno per giorno nel frattempo.
    """
    p = 1 - livello
    z = stats.norm.ppf(p)
    passato = r.shift(1)  # alla data t si conoscono solo i rendimenti fino a t−1
    mobile = passato.rolling(finestra)
    media, dev = mobile.mean(), mobile.std()
    out = pd.DataFrame(index=r.index)
    out["Storico"] = -mobile.quantile(p)
    out["Normale"] = -(media + z * dev)
    asimmetria, curtosi = mobile.skew(), mobile.kurt()
    q_cf = [quantile_cornish_fisher(p, s, k) if np.isfinite(s) else np.nan for s, k in zip(asimmetria, curtosi)]
    out["Cornish-Fisher"] = -(media + dev * np.array(q_cf))
    out["EWMA (RiskMetrics)"] = -z * volatilita_ewma(r).shift(1)
    out["GARCH(1,1)"] = -z * np.sqrt(_varianze_garch_fuori_campione(r.to_numpy(), finestra, ristima_garch))
    return out.iloc[finestra + 1:]


def _varianze_garch_fuori_campione(x: np.ndarray, finestra: int, passo: int) -> np.ndarray:
    s2 = np.full(len(x), np.nan)
    for inizio in range(finestra, len(x), passo):
        stima = x[inizio - finestra:inizio]
        modello: Garch = stima_garch(stima)
        # Filtra dall'inizio della finestra di stima fino alla fine del blocco di previsione,
        # partendo dalla varianza della sola finestra di stima
        fine = min(inizio + passo, len(x))
        v = modello.varianze(x[inizio - finestra:fine - 1], iniziale=float(np.var(stima)))
        s2[inizio:fine] = v[finestra:]
    return s2


def violazioni(r: pd.Series, var: pd.DataFrame) -> pd.DataFrame:
    """Matrice booleana: ``True`` quando la perdita del giorno supera il VaR previsto."""
    return var.apply(lambda col: -r.loc[var.index] > col)


def test_kupiec(n_violazioni: int, n: int, livello: float) -> tuple[float, float]:
    """Test di copertura incondizionata di Kupiec (POF): le violazioni sono quante ci si aspetta?

    Restituisce la statistica del rapporto di verosimiglianza e il p-value (chi quadro con 1 grado di libertà).
    """
    p, x = 1 - livello, n_violazioni
    frequenza = x / n
    ll_nulla = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_alt = (n - x) * np.log(1 - frequenza) if x < n else 0.0
    ll_alt += x * np.log(frequenza) if x > 0 else 0.0
    lr = -2 * (ll_nulla - ll_alt)
    return float(lr), float(stats.chi2.sf(lr, 1))


def test_christoffersen(sequenza: np.ndarray | pd.Series) -> tuple[float, float]:
    """Test di indipendenza di Christoffersen: le violazioni arrivano a grappoli?

    Confronta la probabilità di una violazione dopo un giorno senza violazione e dopo una violazione.
    """
    v = np.asarray(sequenza, dtype=int)
    prec, succ = v[:-1], v[1:]
    n00 = int(np.sum((prec == 0) & (succ == 0)))
    n01 = int(np.sum((prec == 0) & (succ == 1)))
    n10 = int(np.sum((prec == 1) & (succ == 0)))
    n11 = int(np.sum((prec == 1) & (succ == 1)))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(k0: int, k1: int, q: float) -> float:
        return (k0 * np.log(1 - q) if k0 else 0.0) + (k1 * np.log(q) if k1 else 0.0)

    lr = -2 * (ll(n00 + n10, n01 + n11, pi) - ll(n00, n01, pi0) - ll(n10, n11, pi1))
    lr = max(lr, 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def semaforo_basilea(n_violazioni_250: int, livello: float = 0.99) -> str:
    """Zona del semaforo di Basilea sulle ultime 250 giornate.

    Verde finché la probabilità binomiale di osservare al massimo quel numero di violazioni con un modello
    corretto resta sotto il 95%, giallo sotto il 99,99%, rosso oltre: al 99% verde fino a 4, rosso da 10.
    """
    cumulata = stats.binom.cdf(n_violazioni_250, 250, 1 - livello)
    if cumulata < 0.95:
        return "Verde"
    if cumulata < 0.9999:
        return "Giallo"
    return "Rosso"


def riepilogo(r: pd.Series, var: pd.DataFrame, livello: float = 0.99) -> pd.DataFrame:
    """Tabella di convalida: violazioni attese ed effettive, test di Kupiec e Christoffersen, semaforo."""
    v = violazioni(r, var)
    n = len(v)
    righe = {}
    for metodo in var.columns:
        x = int(v[metodo].sum())
        _, p_kupiec = test_kupiec(x, n, livello)
        _, p_indip = test_christoffersen(v[metodo])
        perdite = -r.loc[v.index][v[metodo]]
        righe[metodo] = {
            "Giorni": n,
            "Violazioni attese": n * (1 - livello),
            "Violazioni": x,
            "Frequenza": x / n,
            "p-value Kupiec": p_kupiec,
            "p-value indipendenza": p_indip,
            "Eccesso medio": float((perdite - var.loc[perdite.index, metodo]).mean()) if x else 0.0,
            "Ultimi 250 giorni": int(v[metodo].iloc[-250:].sum()),
            "Semaforo": semaforo_basilea(int(v[metodo].iloc[-250:].sum()), livello),
        }
    return pd.DataFrame(righe).T
