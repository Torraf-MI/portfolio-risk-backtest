"""Prezzi: da file CSV, da Yahoo Finance o dati dimostrativi con crisi di mercato riconoscibili."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Universo dimostrativo: asset class tipiche del portafoglio di un cliente privato.
# Parametri annui (rendimento, volatilità) indicativi, non previsioni.
ASSET_ESEMPIO = {
    "Azionario Globale": (0.075, 0.15),
    "Azionario Europa": (0.065, 0.17),
    "Azionario Emergenti": (0.070, 0.20),
    "Obbligazioni Gov. EUR": (0.022, 0.050),
    "Obbligazioni Corporate EUR": (0.030, 0.050),
    "Oro": (0.055, 0.15),
    "Monetario EUR": (0.012, 0.004),
}

CORRELAZIONE_ESEMPIO = np.array(
    [
        [1.00, 0.88, 0.75, -0.15, 0.30, 0.10, 0.00],
        [0.88, 1.00, 0.70, -0.10, 0.35, 0.05, 0.00],
        [0.75, 0.70, 1.00, -0.10, 0.30, 0.20, 0.00],
        [-0.15, -0.10, -0.10, 1.00, 0.70, 0.25, 0.10],
        [0.30, 0.35, 0.30, 0.70, 1.00, 0.20, 0.10],
        [0.10, 0.05, 0.20, 0.25, 0.20, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.10, 0.10, 0.00, 1.00],
    ]
)

# Portafoglio tipico di un cliente con profilo bilanciato, usato negli esempi.
PESI_ESEMPIO = {
    "Azionario Globale": 0.35,
    "Azionario Europa": 0.10,
    "Azionario Emergenti": 0.05,
    "Obbligazioni Gov. EUR": 0.25,
    "Obbligazioni Corporate EUR": 0.15,
    "Oro": 0.05,
    "Monetario EUR": 0.05,
}


@dataclass(frozen=True)
class Episodio:
    """Periodo di mercato con il rendimento complessivo di ogni asset nella finestra."""

    nome: str
    inizio: str
    fine: str
    rendimenti: dict[str, float]
    volatilita: float = 2.5  # moltiplicatore della volatilità giornaliera nella finestra
    azioni_obbligazioni: float | None = None  # correlazione azioni-governativi, se diversa dal normale
    stress: bool = True  # False per le fasi di recupero, che non sono scenari di stress


# Le crisi dei dati dimostrativi ricalcano per date e ordine di grandezza episodi reali,
# ma i numeri sono simulati e non sono quelli storici degli indici.
EPISODI = [
    Episodio("Crisi finanziaria 2008", "2007-10-10", "2009-03-09",
             {"Azionario Globale": -0.50, "Azionario Europa": -0.57, "Azionario Emergenti": -0.60,
              "Obbligazioni Gov. EUR": 0.09, "Obbligazioni Corporate EUR": -0.07, "Oro": 0.22,
              "Monetario EUR": 0.05}, volatilita=2.2, azioni_obbligazioni=-0.35),
    Episodio("Ripresa 2009", "2009-03-10", "2010-04-30",
             {"Azionario Globale": 0.62, "Azionario Europa": 0.70, "Azionario Emergenti": 0.95,
              "Obbligazioni Gov. EUR": 0.04, "Obbligazioni Corporate EUR": 0.16, "Oro": 0.25,
              "Monetario EUR": 0.01}, volatilita=1.4, stress=False),
    Episodio("Crisi del debito sovrano 2011", "2011-07-01", "2011-10-04",
             {"Azionario Globale": -0.17, "Azionario Europa": -0.27, "Azionario Emergenti": -0.24,
              "Obbligazioni Gov. EUR": -0.02, "Obbligazioni Corporate EUR": -0.03, "Oro": 0.14,
              "Monetario EUR": 0.003}, volatilita=2.3),
    Episodio("Covid 2020", "2020-02-19", "2020-03-23",
             {"Azionario Globale": -0.33, "Azionario Europa": -0.37, "Azionario Emergenti": -0.31,
              "Obbligazioni Gov. EUR": -0.01, "Obbligazioni Corporate EUR": -0.07, "Oro": -0.03,
              "Monetario EUR": 0.0}, volatilita=4.0),
    Episodio("Ripresa 2020", "2020-03-24", "2020-12-31",
             {"Azionario Globale": 0.52, "Azionario Europa": 0.42, "Azionario Emergenti": 0.58,
              "Obbligazioni Gov. EUR": 0.03, "Obbligazioni Corporate EUR": 0.08, "Oro": 0.10,
              "Monetario EUR": -0.003}, volatilita=1.3, stress=False),
    Episodio("Rialzo dei tassi 2022", "2022-01-03", "2022-10-12",
             {"Azionario Globale": -0.19, "Azionario Europa": -0.22, "Azionario Emergenti": -0.27,
              "Obbligazioni Gov. EUR": -0.17, "Obbligazioni Corporate EUR": -0.16, "Oro": -0.02,
              "Monetario EUR": -0.002}, volatilita=1.6, azioni_obbligazioni=0.55),
]

# Riconoscimento della classe di un asset dal nome (italiano o inglese, ticker comuni)
_CLASSI = [
    ("liquidita", r"monetari|liquidit|cash|money market|xeon|\bmm\b|conto"),
    ("oro", r"\boro\b|gold|sgld|phau|metall"),
    ("corporate", r"corporate|corp|high yield|credit|ieac"),
    ("governativi", r"gov|btp|bund|treasur|tesoro|stato|aggregate|bond|obblig|xgle|agg"),
    ("azioni", r"azion|equity|stock|msci|s&p|ftse|stoxx|vwce|swda|iwda|eimi|csspx|world|emerg|index|indice"),
]


def classe(nome: str) -> str | None:
    """Classe di attività dedotta dal nome: azioni, governativi, corporate, oro, liquidita o None."""
    testo = nome.lower()
    for c, schema in _CLASSI:
        if re.search(schema, testo):
            return c
    return None


def load_csv(path: str | Path, date_column: str | None = None) -> pd.DataFrame:
    """Legge un CSV di prezzi: una colonna data e una colonna per ogni asset.

    È lo stesso formato usato da portfolio-optimizer e dalle esportazioni di wealth-dashboard.
    """
    df = pd.read_csv(path)
    date_column = date_column or df.columns[0]
    df[date_column] = pd.to_datetime(df[date_column])
    return df.set_index(date_column).sort_index().astype(float).ffill().dropna()


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Scarica prezzi di chiusura rettificati da Yahoo Finance (richiede ``yfinance``)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dipende dall'ambiente
        raise ImportError("Installa yfinance: pip install 'portfolio-risk-backtest[yahoo]'") from exc

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError("Nessun dato scaricato: verifica i ticker e la connessione")
    prices = data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices.dropna(how="all").ffill().dropna()


def _correlazione(azioni_obbligazioni: float | None) -> np.ndarray:
    c = CORRELAZIONE_ESEMPIO.copy()
    if azioni_obbligazioni is not None:
        c[:3, 3] = c[3, :3] = azioni_obbligazioni
        c[:3, 4] = c[4, :3] = max(azioni_obbligazioni, c[0, 4]) + 0.1
    # Riporta la matrice a definita positiva se la modifica l'ha resa incoerente
    valori, vettori = np.linalg.eigh(c)
    c = vettori @ np.diag(np.clip(valori, 1e-4, None)) @ vettori.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def generate_sample_prices(
    start: str = "2007-01-01",
    end: str = "2025-12-31",
    seed: int = 3,
) -> pd.DataFrame:
    """Genera prezzi sintetici giornalieri con volatilità variabile nel tempo e crisi di mercato.

    La volatilità segue un processo persistente (fasi calme e fasi agitate, come nei mercati reali), e
    nelle finestre di :data:`EPISODI` il rendimento complessivo di ogni asset è esattamente quello indicato.
    Servono a provare il progetto senza connessione: i numeri non rappresentano l'andamento reale di
    alcuno strumento.
    """
    rng = np.random.default_rng(seed)
    date = pd.bdate_range(start, end)
    n, k = len(date), len(ASSET_ESEMPIO)
    mu = np.array([m for m, _ in ASSET_ESEMPIO.values()])
    vol = np.array([v for _, v in ASSET_ESEMPIO.values()])

    # Volatilità stocastica comune: log-volatilità AR(1) normalizzata a varianza media unitaria
    h = np.zeros(n)
    eps = rng.normal(0, 0.07, n)
    for t in range(1, n):
        h[t] = 0.98 * h[t - 1] + eps[t]
    scala = np.exp(h)
    scala /= np.sqrt(np.mean(scala**2))

    z = rng.multivariate_normal(np.zeros(k), _correlazione(None), size=n)
    # Code grasse: qualche giornata di shock comune
    z[rng.random(n) < 0.003] *= 2.5
    r = z * scala[:, None]

    fuori = np.ones(n, dtype=bool)
    for ep in EPISODI:
        finestra = (date >= ep.inizio) & (date <= ep.fine)
        m = int(finestra.sum())
        if m < 2:
            continue
        fuori &= ~finestra
        zz = rng.multivariate_normal(np.zeros(k), _correlazione(ep.azioni_obbligazioni), size=m)
        rr = zz * vol / np.sqrt(252) * ep.volatilita
        obiettivo = np.log1p(np.array([ep.rendimenti[a] for a in ASSET_ESEMPIO]))
        # Ponte: rumore a media nulla nella finestra più una deriva che porta al rendimento indicato
        r[finestra] = rr - rr.mean(axis=0) + obiettivo / m

    # Fuori dalle crisi media e volatilità sono calibrate in modo che sull'intero periodo ogni asset abbia
    # il rendimento composto e la volatilità indicati, qualunque sia il seme casuale
    n_fuori = int(fuori.sum())
    in_crisi = r[~fuori]
    var_fuori = (vol**2 / 252 * n - (in_crisi**2).sum(axis=0)) / n_fuori
    vol_fuori = np.sqrt(np.maximum(var_fuori, (0.5 * vol) ** 2 / 252))
    deriva = (np.log1p(mu) * n / 252 - in_crisi.sum(axis=0)) / n_fuori
    r[fuori] = (r[fuori] - r[fuori].mean(axis=0)) / r[fuori].std(axis=0) * vol_fuori + deriva

    livelli = 100 * np.exp(np.vstack([np.zeros(k), np.cumsum(r[1:], axis=0)]))
    return pd.DataFrame(livelli, index=date, columns=list(ASSET_ESEMPIO)).round(4)


def parse_pesi(testo: str, colonne: list[str]) -> pd.Series:
    """Legge pesi nel formato ``"Nome=0.4, Altro=0.6"`` (nomi parziali ammessi se univoci)."""
    pesi = {}
    for parte in filter(None, (p.strip() for p in testo.split(","))):
        if "=" not in parte:
            raise ValueError(f"Peso non valido: {parte!r} (usa Nome=0.4)")
        nome, valore = (s.strip() for s in parte.rsplit("=", 1))
        trovati = [c for c in colonne if c == nome] or [c for c in colonne if nome.lower() in c.lower()]
        if len(trovati) != 1:
            raise ValueError(f"Asset {nome!r} non trovato o ambiguo tra: {', '.join(colonne)}")
        pesi[trovati[0]] = float(valore.rstrip("%")) / (100 if valore.endswith("%") else 1)
    return normalizza_pesi(pd.Series(pesi), colonne)


def normalizza_pesi(pesi: pd.Series | dict, colonne: list[str]) -> pd.Series:
    """Allinea i pesi alle colonne dei prezzi e verifica che sommino a 1."""
    s = pd.Series(pesi, dtype=float).reindex(colonne).fillna(0.0)
    if (s < 0).any():
        raise ValueError("Sono ammessi solo pesi positivi (niente vendite allo scoperto)")
    if not np.isclose(s.sum(), 1.0, atol=1e-3):
        raise ValueError(f"I pesi devono sommare a 1 (ora {s.sum():.3f})")
    return s / s.sum()
