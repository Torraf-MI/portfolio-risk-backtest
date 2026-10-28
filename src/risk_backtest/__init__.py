"""Analisi del rischio e backtest: VaR/CVaR, convalida dei modelli, stress test e confronto tra strategie."""

from .backtest import RisultatoBacktest, confronta, esegui
from .convalida import previsioni_var, riepilogo, test_christoffersen, test_kupiec
from .dati import generate_sample_prices, load_csv, normalizza_pesi, parse_pesi
from .metriche import alfa_beta, drawdown, sharpe_probabilistico, sintesi
from .rischio import (
    contributi_cvar,
    contributi_volatilita,
    cvar_storico,
    rendimenti,
    rendimenti_portafoglio,
    stima_garch,
    tabella_var,
    var_storico,
    volatilita_ewma,
)
from .strategie import risk_parity, strategie_standard
from .stress import SCENARI_IPOTETICI, Scenario, stress_test

__all__ = [
    "SCENARI_IPOTETICI",
    "RisultatoBacktest",
    "Scenario",
    "alfa_beta",
    "confronta",
    "contributi_cvar",
    "contributi_volatilita",
    "cvar_storico",
    "drawdown",
    "esegui",
    "generate_sample_prices",
    "load_csv",
    "normalizza_pesi",
    "parse_pesi",
    "previsioni_var",
    "rendimenti",
    "rendimenti_portafoglio",
    "riepilogo",
    "risk_parity",
    "sharpe_probabilistico",
    "sintesi",
    "stima_garch",
    "strategie_standard",
    "stress_test",
    "tabella_var",
    "test_christoffersen",
    "test_kupiec",
    "var_storico",
    "volatilita_ewma",
]

__version__ = "0.1.0"
