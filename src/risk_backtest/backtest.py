"""Motore di backtest: ribilanciamento periodico, deriva dei pesi tra un ribilanciamento e l'altro, costi.

Le decisioni prese alla chiusura di una data si applicano dal giorno successivo, quindi nessuna strategia
usa informazioni future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategie import Strategia


@dataclass
class RisultatoBacktest:
    nome: str
    rendimenti: pd.Series  # rendimenti giornalieri netti di costi
    pesi: pd.DataFrame  # pesi obiettivo a ogni ribilanciamento
    rotazione: pd.Series  # quota di portafoglio scambiata a ogni ribilanciamento

    @property
    def valore(self) -> pd.Series:
        return (1 + self.rendimenti).cumprod()

    @property
    def rotazione_annua(self) -> float:
        anni = len(self.rendimenti) / 252
        return float(self.rotazione.sum() / anni) if anni else 0.0


def date_ribilanciamento(indice: pd.DatetimeIndex, frequenza: str = "ME") -> pd.DatetimeIndex:
    """Ultimo giorno di borsa di ogni periodo (``ME`` mese, ``QE`` trimestre, ``YE`` anno)."""
    serie = pd.Series(indice, index=indice)
    return pd.DatetimeIndex(serie.groupby(indice.to_period(frequenza[0])).max().to_numpy())


def esegui(
    prezzi: pd.DataFrame,
    strategia: Strategia,
    nome: str = "Strategia",
    inizio: str | pd.Timestamp | None = None,
    frequenza: str = "ME",
    costi_bps: float = 10.0,
    storico_minimo: int = 252,
) -> RisultatoBacktest:
    """Simula la strategia giorno per giorno.

    ``inizio`` è la prima data di investimento (per default dopo ``storico_minimo`` giorni di dati, che
    servono alle strategie per stimare rischio e tendenza). Il costo, in punti base sul controvalore
    scambiato, è dedotto il giorno del ribilanciamento.
    """
    r = prezzi.pct_change().fillna(0.0).to_numpy()
    date = prezzi.index
    primo = storico_minimo if inizio is None else max(int(date.searchsorted(pd.Timestamp(inizio))), storico_minimo)
    if primo >= len(date) - 1:
        raise ValueError("Storico insufficiente per il backtest")
    ribilancia = set(date_ribilanciamento(date[primo:], frequenza))
    ribilancia.add(date[primo - 1])  # investimento iniziale alla chiusura del giorno prima

    w = np.zeros(prezzi.shape[1])
    netti = np.zeros(len(date))
    pesi, rotazione = {}, {}
    costo = costi_bps / 10_000
    costo_iniziale = 0.0
    for t in range(primo - 1, len(date)):
        if t >= primo:
            crescita = w * (1 + r[t])
            lordo = crescita.sum() + (1 - w.sum())  # la liquidità residua rende zero
            w = crescita / lordo
            netti[t] = lordo * (1 - costo_iniziale) - 1
            costo_iniziale = 0.0
        if date[t] in ribilancia and t < len(date) - 1:
            obiettivo = strategia(prezzi.iloc[: t + 1]).reindex(prezzi.columns).fillna(0.0).to_numpy()
            scambio = float(np.abs(obiettivo - w).sum())
            if t >= primo:
                netti[t] = (1 + netti[t]) * (1 - scambio * costo) - 1
            else:
                costo_iniziale = scambio * costo  # acquisto iniziale, dedotto il primo giorno
            w = obiettivo
            pesi[date[t]] = obiettivo
            rotazione[date[t]] = scambio
    serie = pd.Series(netti[primo:], index=date[primo:], name=nome)
    return RisultatoBacktest(
        nome=nome,
        rendimenti=serie,
        pesi=pd.DataFrame(pesi, index=prezzi.columns).T,
        rotazione=pd.Series(rotazione, name="Rotazione"),
    )


def confronta(
    prezzi: pd.DataFrame,
    strategie: dict[str, Strategia],
    **kwargs,
) -> dict[str, RisultatoBacktest]:
    """Esegue tutte le strategie sullo stesso periodo e con gli stessi costi."""
    return {nome: esegui(prezzi, s, nome=nome, **kwargs) for nome, s in strategie.items()}
